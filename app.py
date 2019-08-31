
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import io
import base64
import dill
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from wordcloud import WordCloud 
from flask import Flask, render_template, request


# In[2]:


def make_pred(tfidf_transformer,lgbm_model,nn_model,tfidf2_dict,note_list):
    lgbm_input=tfidf_transformer.transform(note_list)
    lgbm_prob_output=lgbm_model.predict_proba(lgbm_input)[0, 1]
    lgbm_contri_output=lgbm_model.predict_proba(lgbm_input,pred_contrib=True).tolist()[0]
    nn_unpadded_input=[[tfidf2_dict[word] for word in note.split() if word in tfidf2_dict.keys()] for note in note_list]
    maxlen=2000
    nn_padded_input=pad_sequences(nn_unpadded_input, dtype='float32', padding='post', maxlen=maxlen)
    nn_output=nn_model.predict(nn_padded_input)[0,0]
    
    return lgbm_prob_output,nn_output,lgbm_contri_output


# In[3]:


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


# In[4]:


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
                         figsize=(6,10))
    my_plot.plot(step.index, step.values,'k')
    
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


# In[5]:


app = Flask(__name__)


# In[6]:


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
    if result_list==[]:
        result_list.append(('empty',None,None))
            
    return render_template('results.html',
                           result_list=result_list)


# In[7]:


if __name__ == '__main__':
    app.run(port=4799)


# In[ ]:


"Admission Date:                Discharge Date:   Date of Birth:               Sex:   MService: CARDIOTHORACICAllergies:CodeineAttending:Chief Complaint:Coronary artery diseaseMajor Surgical or Invasive Procedure:CABG X 5 LIMA > LAD, SVG>PLV>PDA, SVG>OM1>OM2History of Present Illness:Mr.  is a 56 yo male with coronary artery disease, whopresented to  on  for an elective CABG.Past Medical History:CAD/MI ()HTNHyerlipidemiaDMPhysical Exam:Alert, oriented, well-nourished, comfortableChest clear bilaterallyRRRAbdomen soft, nontenderExtremities warm, well-perfusedBrief Hospital Course:Mr.   CABG x5 on , which he tolerated well(see Op Note).  He was transferred to the cardiac intensive careunit post-operatively, as per routine.  He was extubated soonthereafter. It should be noted that this patient is a difficultintubation, but with the close assistance of anesthesia, therewere no issues with extubation. He would remain in stablecondition throughout his hospital stay.  His chest tubes wereremoved on POD 1, and he was transferred from the ICU to thefloor in stable condtion.  He was soon ambulating easily andoften on his own.  On POD 3, he did have a fever, however, hecontinued feeling well.  His wound showed no signs of infection. A chest x-ray revealed no abnormal findings.  His WBC on POD 4was within normal limits.  On POD 5, with Mr.  feelingwell, ambulating easily, his sternum stable, and with his woundappearing to be healing well, he was discharged to home in goodcondition.  He will follow-up within the next month forpost-operative evaluation with Dr. .Discharge Medications:1. Potassium Chloride 10 mEq Capsule, Sustained Release Sig: Two(2) Capsule, Sustained Release PO Q12H (every 12 hours) for 5days.Disp:*20 Capsule, Sustained Release(s)* Refills:*0*2. Aspirin 81 mg Tablet, Delayed Release (E.C.) Sig: One (1)Tablet, Delayed Release (E.C.) PO DAILY (Daily).Disp:*30 Tablet, Delayed Release (E.C.)(s)* Refills:*2*3. Ranitidine HCl 150 mg Tablet Sig: One (1) Tablet PO BID (2times a day).Disp:*60 Tablet(s)* Refills:*2*4. Oxycodone-Acetaminophen 5-325 mg Tablet Sig: 1-2 Tablets POevery 4-6 hours as needed for pain.Disp:*50 Tablet(s)* Refills:*0*5. Simvastatin 40 mg Tablet Sig: Two (2) Tablet PO DAILY(Daily).Disp:*60 Tablet(s)* Refills:*2*6. Clopidogrel 75 mg Tablet Sig: One (1) Tablet PO DAILY(Daily).Disp:*30 Tablet(s)* Refills:*2*7. Metoprolol Tartrate 25 mg Tablet Sig: One (1) Tablet PO BID(2 times a day).Disp:*60 Tablet(s)* Refills:*2*8. Furosemide 20 mg Tablet Sig: One (1) Tablet PO BID (2 times aday) for 5 days.Disp:*10 Tablet(s)* Refills:*0*9. Glyburide 5 mg Tablet Sig: Two (2) Tablet PO DAILY (Daily).Disp:*60 Tablet(s)* Refills:*2*10. Metformin 850 mg Tablet Sig: One (1) Tablet PO DAILY(Daily).Disp:*30 Tablet(s)* Refills:*2*Discharge Disposition:Home With ServiceFacility: VNADischarge Diagnosis:CADDMHTNDischarge Condition:goodDischarge Instructions:no lifting > 10# for 10 weeksmay shower, no bathing or swimming for 1 monthno creams, lotions or powders to any incisionsFollowup Instructions:with Dr.  in  weekswith Dr.   weekswith Dr.  in 4 weeks PATIENT/TEST INFORMATION:Indication: Abnormal ECG. Chest pain. Coronary artery disease. Hypertension.Status: InpatientDate/Time:  at 09:24Test: TEE (Complete)Doppler: Full Doppler and color DopplerContrast: NoneTechnical Quality: AdequateINTERPRETATION:Findings:LEFT ATRIUM: Normal LA size. No spontaneous echo contrast in the body of theLA. No mass/thrombus in the   LAA. No spontaneous echo contrast is seen inthe LAA. Good (>20 cm/s) LAA ejection velocity. No thrombus in the LAA.RIGHT ATRIUM/INTERATRIAL SEPTUM: Normal RA size. No spontaneous echo contrastin the body of the RA. A catheter or pacing wire is seen in the RA andextending into the RV. No spontaneous echo contrast in the RAA. No thrombus inthe RAA. Normal interatrial septum. No ASD by 2D or color Doppler. The IVC isnormal in diameter with appropriate phasic respirator variation.LEFT VENTRICLE: Wall thickness and cavity dimensions were obtained from 2Dimages. Normal LV wall thicknesses and cavity size. Normal LV wall thickness.Normal LV cavity size. No LV aneurysm. Mild regional LV systolic dysfunction.Mildly depressed LVEF. No LV mass/thrombus.LV WALL MOTION: Regional LV wall motion abnormalities include: basal anterior- normal; mid anterior - hypo; basal anteroseptal - normal; mid anteroseptal -hypo; basal inferoseptal - normal; mid inferoseptal - normal; basal inferior -normal; mid inferior - normal; basal inferolateral - normal; mid inferolateral- normal; basal anterolateral - normal; mid anterolateral - normal; anteriorapex - hypo; septal apex - hypo; inferior apex - normal; lateral apex -normal; apex - hypo;RIGHT VENTRICLE: Normal RV chamber size and free wall motion.AORTA: Normal aortic root diameter. Simple atheroma in aortic root. Normalascending aorta diameter. Simple atheroma in ascending aorta. Normal aorticarch diameter. Simple atheroma in aortic arch. Normal descending aortadiameter. Simple atheroma in descending aorta.AORTIC VALVE: Mildly thickened aortic valve leaflets. No masses or vegetationson aortic valve. Filamentous strands on the aortic leaflets c/with Lambl'sexcresences (normal variant). No AS. Trace AR.MITRAL VALVE: Mildly thickened mitral valve leaflets. No mass or vegetation onmitral valve. Mild mitral annular calcification. Calcified tips of papillarymuscles. No MS. Trivial MR.TRICUSPID VALVE: Normal tricuspid valve leaflets with trivial TR.PULMONIC VALVE/PULMONARY ARTERY: Normal pulmonic valve leaflets withphysiologic PR.PERICARDIUM: No pericardial effusion.GENERAL COMMENTS: A TEE was performed in the location listed above. I certifyI was present in compliance with HCFA regulations. No TEE relatedcomplications. The patient received antibiotic prophylaxis. The TEE probe waspassed with assistance from the anesthesioology staff using a laryngoscope.The patient was under general anesthesia throughout the procedure.Conclusions:PRE-CPB: The left atrium is normal in size. No spontaneous echo contrast isseen in the body of the left atrium. No mass/thrombus is seen in the leftatrium or left atrial appendage. No spontaneous echo contrast is seen in theleft atrial appendage. No thrombus is seen in the left atrial appendage. Nospontaneous echo contrast is seen in the body of the right atrium. No thrombusis seen in the right atrial appendage No atrial septal defect is seen by 2D orcolor Doppler. Left ventricular wall thicknesses and cavity size are normal.Left ventricular wall thicknesses are normal. The left ventricular cavity sizeis normal. No left ventricular aneurysm is seen. There is mild regional leftventricular systolic dysfunction. Overall left ventricular systolic functionis mildly depressed. No masses or thrombi are seen in the left ventricle.Resting regional wall motion abnormalities include apical, anterior andanterior septal hypokinesis.. Right ventricular chamber size and free wallmotion are normal. There are simple atheroma in the aortic root. There aresimple atheroma in the ascending aorta. There are simple atheroma in theaortic arch. There are simple atheroma in the descending thoracic aorta. Theaortic valve leaflets are mildly thickened. No masses or vegetations are seenon the aortic valve. There are filamentous strands on the aortic leafletsconsistent with Lambl's excresences (normal variant). There is no aortic valvestenosis. Trace aortic regurgitation is seen. The mitral valve leaflets aremildly thickened. No mass or vegetation is seen on the mitral valve. Trivialmitral regurgitation is seen. There is no pericardial effusion.POST-CPB: Biventricular systolic function is preserved with LVEF 40-45%. RWMAas described. Trqace MR,TR,PI,AI as pre-cpb. On phenylephrine infusion. Sinus rhythm. Small Q waves in the inferior leads consistent with possibleprior inferior wall myocardial infarction. There is a late transitionconsistent with possible prior anterior wall myocardial infarction. Compared tothe previous tracing no significant change. Narrative NoteAdmit from OR, s/p CABG x 5. Extremely difficult intubation. 45 min. to fiberoptically intubate.  Will need anesthesia and MD   for tube removal. Return to CSRU A- paced. A wires inappropriately firing, A output wire switched to Vport, incr to 25 output, with good capture.  Underlying rhythm, slow sinus.NEURO: Propofol off, awake to verbal. Follows commands, PERRLA. Given toradol for perceived pain. Attempt to wean to extubation.CARDIO: HR, 70's , NSR. no ectopy. BP stable, neo gtt weaned to off.2 A , 2 V wires attached. Ademand 60 set as backup, A output wire in V-port. CT's with minimal drainage.  Sternal and mediastinal dsgs,CDI. RIJ with PA cath placed, waveform sharp. CI 2.10. 500cc bolus fluid given. R radial a-line placed, waveform sharp. Pedal pulses, weak but palpable.RESP:LS clear, dim. bilaterally. remains intubated. Reversals given and propofol off.  Unable to lift head off pillow at this time. ABG's show fluid deficit,  fluids infusing.ENDO: Insulin gtt per CSRU protocol.SOCiAL: Friend at , updated on surgery and ICU orientation.PLAN: hemodynamic monitoring, pulmonary toilet, may have to keep intubated until a.m, mobilize, treat and D/c to the floor. CSRU NPNNEURO: A/O X3 PLEASANT. MAE TO COMMAND & SPONTANEOUSLY. AFEBRILE. TORADOL IM FOR PAIN.CV: SR, RARE PAC'S. A/V WIRES INTACT AND FUNCTIONING. NEO OFF/ON TO MAINTAIN SBP >90. FLUID BOLUSES GIVEN FOR LOW MIXED VENOUS O2 SAT, LOW CVP, LOW PAD, LOW B/P WITH GOOD RESULTS - SEE CAREVUE. CO/CI BY THERMODILUTION CORRELATES WITH FICK. CT WITH S/S DRAINAGE.RESP: EXTUBATED AT 1900, WEANED TO 2LNC WITHOUT DIFFICULTY. LUNGS CLEAR WITH DIM BASES. INSTRUCTED ON USE OF COUGH PILLOW AND I/S. I/S TO . NO COUGH/CONGESTION. PT HAS AUDIBLE SNORE WHEN SLEEPING. SPO2 >98%.GI: TOLERATING CLEAR LIQUIDS. HYPO BS, R>L. NO FLATUS, NO BM.GU: FOLEY PATENT. HUO ADEQUATE.ENDO: INSULIN GTT PER CSRU PROTOCOL. TO RESTART ORAL MEDS WHEN TAKING IN FULL MEALS.SOCIAL: CALL RECEIVED FROM ROOM , . UPDATED AS TO STATUS.PLAN: WEAN INSULIN GTT & START ON SLIDING SCALE. PULMONARY HYGEINE. OOB > CHAIR IF OK WITH TEAM. NEO OFF AT THIS TIME."


# In[ ]:


a=[]


# In[ ]:


a is None

