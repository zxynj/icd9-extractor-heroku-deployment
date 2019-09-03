
# coding: utf-8

# In[8]:


import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import io
import base64
import dill
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from wordcloud import WordCloud 
from flask import Flask, render_template, request


# In[9]:


def make_pred(tfidf_transformer,lgbm_model,nn_model,tfidf2_dict,note_list):
    lgbm_input=tfidf_transformer.transform(note_list)
    lgbm_prob_output=lgbm_model.predict_proba(lgbm_input)[0, 1]
    lgbm_contri_output=lgbm_model.predict_proba(lgbm_input,pred_contrib=True).tolist()[0]
    nn_unpadded_input=[[tfidf2_dict[word] for word in note.split() if word in tfidf2_dict.keys()] for note in note_list]
    maxlen=2000
    nn_padded_input=pad_sequences(nn_unpadded_input, dtype='float32', padding='post', maxlen=maxlen)
    nn_output=nn_model.predict(nn_padded_input)[0,0]
    
    return lgbm_prob_output,nn_output,lgbm_contri_output


# In[10]:


def build_word_cloud(importance_dict):
    wordcloud = WordCloud(width = 800, height = 800, 
                          background_color ='white', 
                          min_font_size = 10).generate_from_frequencies(importance_dict) 
    
    img = io.BytesIO()
    
    plt.figure(figsize = (6,6), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


# In[11]:


def build_waterfall(prob_list,prob_source_name_list):
    trans = pd.DataFrame(data={'Importance': prob_list},index=prob_source_name_list).sort_values(by=['Importance'])
    blank=trans.Importance.cumsum().shift(1).fillna(0)
    total = trans.sum().Importance
    trans.loc["net"] = -total
    blank.loc["net"] = total
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan
    
    img = io.BytesIO()
    my_plot = trans.plot(kind='bar',
                         stacked=True,
                         bottom=blank,legend=None,
                         title="Predicted Probability Waterfall Chart",
                         ylim=(-0.5,1.5),
                         figsize=(6,6))
    my_plot.plot(step.index, step.values,'k')
    
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


# In[12]:


app = Flask(__name__)


# In[13]:


@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/analysis')
def analysispage():
    return render_template('analysis.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results', methods=['POST', 'GET'])
def resultspage():
    try:
        note = request.form["note_input"]
    except:
        return render_template("analysis.html")
    if note.strip()=='' or note==None:
        return render_template("analysis.html")
    
    threshhold=0.5
    icd9_list=['4019']
    result_list=[]
    for icd9 in icd9_list:
        tfidf2_dict = dill.load(open('data/tfidf2_dict.dict', 'rb'))
        tfidf_transformer = dill.load(open('model/'+icd9+'_tfidf_transformer.fitted', 'rb'))
        lgbm_model = dill.load(open('model/'+icd9+'_lgbm_model.fitted', 'rb'))
        nn_model=load_model('model/'+icd9+'_nn_model_fitted.h5')  
    
        lgbm_prob_output,nn_output,lgbm_contri_output=make_pred(tfidf_transformer,lgbm_model,nn_model,tfidf2_dict,[note])
    
        if lgbm_prob_output+nn_output>threshhold:
            lgbm_importance_list=lgbm_model.feature_importances_.tolist()
            lgbm_min_importance=min([i for i in lgbm_importance_list if i!=0])/10
            lgbm_importance_list_fixed=[lgbm_min_importance if i==0 else i for i in lgbm_importance_list]
            lgbm_feature_name_list=tfidf_transformer.get_feature_names()
            importance_dict=dict((i,j) for (i,j) in zip(lgbm_feature_name_list, lgbm_importance_list_fixed) if j!=0)
            graph1_url = build_word_cloud(importance_dict)

            prob_cum_list=list(1/(1+np.exp(-np.cumsum(lgbm_contri_output))))
            prob_cum_list.insert(0,0)
            prob_list=list(np.diff(prob_cum_list))
            prob_list.append(min(max(prob_cum_list[-1]+nn_output,0),1)-prob_cum_list[-1])
            prob_source_name_list=['Intercept']
            prob_source_name_list.extend(lgbm_feature_name_list)
            prob_source_name_list.append('convolutional neural network')
            graph2_url = build_waterfall(prob_list,prob_source_name_list)
            
            result_list.append((icd9+': '+max(lgbm_feature_name_list)+' is a potential ICD-9 with probability '+str(sum(prob_list))+'.',
                           graph1_url,graph2_url))
    del note
    
    if result_list==[]:
        result_list.append(('empty',None,None))
            
    return render_template('results.html',
                           result_list=result_list)


# In[14]:


if __name__ == '__main__':
    app.run(port=4799)

