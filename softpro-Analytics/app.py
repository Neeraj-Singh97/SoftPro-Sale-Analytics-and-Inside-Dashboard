git import os
import io
from pyexpat import model
import tempfile
from datetime import datetime
from dateutil import parser
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline 
import plotly.express as px

#sckit learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from vosk import Model, KaldiRecognizer


#set the Environment variable:-
os.environ["path"]+=os.pathsep+ r"C:\Users\asus\Desktop\NeerajSingh-Ai-Ml\softpro-Analytics\ffmpeg\bin"
#C:\Users\asus\OneDrive\Desktop\NeerajSingh\softpro-Analytic\ffmpeg\bin

#Exception model: import:-when open source model not found
try:
    import whisper
except ImportError:
    whisper = None
try:
    from vosk import Model as vosk_model,KaldiRecognizer
    import wave
except ImportError:
    vosk_model = None
    KaldiRecognizer = None
#NLP pipeline
# from transformers import pipeline 
#streamlit UI Design
st.set_page_config(page_title="SoftPro Analytics", page_icon="microphone", layout="wide")
st.title("SoftPro Sales and Analytics inside Dashboard")
st.caption("Audio --> log --> transcript --> Inside --> Recommendation")
# Sidebar for file upload
st.sidebar.header("Settings")
asr_engine = st.sidebar.selectbox("Select ASR Engine",["Whisper","vosk","google","Other"])
if asr_engine == "whisper":
    whisper_size = st.sidebar.selectbox("Select Whisper Model Size", ["tiny", "base", "small", "medium", "large"], index=0)
elif asr_engine == "vosk":
    vosk_model_dir = st.sidebar.text_input("Vosk Model Directory", value="")
else:
    other_asr_option = st.sidebar.text_input("Other ASR Option", value="")
st.sidebar.markdown("_________")
st.sidebar.write("**Export**")
save_intermediate = st.sidebar.checkbox("Save Processed Audio Csv",value=True)
st_intermediate = st.sidebar.success("Intermediate results saved successfully!")
#utilites or Header : Custom function.....
#whisper loading model
@st.cache_resource(show_spinner=False,)
def load_whisper (model_size):
    if whisper is None:
        raise RuntimeError("Whisper ASR engine is not available.")
    model = whisper.load_model(model_size)
    return whisper.load_model(model_size)
#whisper vosk model.
@st.cache_resource(show_spinner=False,)
def load_vosk_model(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError("Invalid Vosk model directory.")
    if vosk_model is None:
        raise RuntimeError("Vosk ASR engine is not available.please install vosk.")
    return vosk_model(model_dir)
#High Frequency pipeline to be loaded
@st.cache_resource(show_spinner=False,)
def load_hf_pipeline():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english") # type: ignore
#traing model. test_split_technique.
@st.cache_resource(show_spinner=False)
def train_sklearn_sentiment(texts:pd.Series, labels:pd.Series):
    #Most + ve , Most -Ve Most Avg.
    y = labels.astype(str).str.lower().replace({'positive': 'pos', 'negative': 'neg', 'neutral': 'neu','n':'negative','p':'positive'})
    x_train, x_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, stop_words='english')
    xtr = vectorizer.fit_transform(x_train)
    xte = vectorizer.transform(x_test)
    #Logistics Regression.
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(xtr, y_train)
    clf.predict(xte)
    #prediction.
    y_pred = clf.predict(xte)
    #report
    report = classification_report(y_test, y_pred,output_dict=False, zero_division=0)
    return vectorizer,clf, report
#distilbert/distilbert-base-uncased-finetuned-sst-2-english
#open source distribution based uncased set-2 english model
#Apply transformer model
#sentiment analysis best model.
#Date Handling
@st.cache_resource(show_spinner=False)
def safe_parse_date(x):
    if pd.isna(x):
        return None
    try:
        return parser.parse(str(x),dayfirst=False,yearfirst=True)
    except Exception as e:
        return None
#Application main Part: transcribe>
@st.cache_resource(show_spinner=False)
def transcribe_with_whisper(audio_bytes:bytes,model,filename:str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".mp3" or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name     
    try:
        result =model.transcribe(path)
        return result.get("text","").strip()
    finally:
        try:
            os.remove(path)
        except Exception as e:
            print("Error cleaning up temporary file:", e)

        return model.transcribe(tmp.name)
#transcribe logic for vosk .
@st.cache_resource(show_spinner=False)
def transcribe_with_vosk(audio_bytes:bytes,model,filename:str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".mp3" or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        if not path.lower().endswith(('.wav', '.mp3',)):
            return "[vosk] please Upload the wav(18kHz) or mp3 file."
        if not path.lower().endswith('.mp3'):
            return "[vosk] please Upload the mp3 (16kHz) file."
        wf = wave.open(path, "wb")

        if wf.getnchannels()!=1 or wf.getsampwidth()!=2:
            return "[vosk] please Upload the wav(18kHz) or mp3 audio file or use whisper."
        if KaldiRecognizer is None:
            return "[vosk] KaldiRecognizer is not available. Please install vosk."
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        text_pieces = []
        while True:
            data = wf.readframes(4000) # type: ignore
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = rec.result() # type: ignore
                text_pieces.append(res)
        final = rec.finalResult() # type: ignore
        text_pieces.append(final)
        #the piece will be joined with a space
        return " ".join(text_pieces)
    except Exception as e:
        return f'[vosk Error] {e}'
    finally:
        try:
            os.remove(path)
        except Exception:
            print("Error cleaning up temporary file:And Vosk Removal Error")
#=======================UI Part=========================
st.title("Transcription")
st.subheader('Step 1:Upload Data')
col1 ,col2 = st.columns(2)
with col1:
    audio_files = st.file_uploader("Upload Audio File call logs(.mp3/.wav)", type=["wav", "mp3"],
    accept_multiple_files=True)
with col2:
    csv_file = st.file_uploader("Upload CSV File log (remarks,students,years,tech stack ,loacation,Date Optinal label)", type=["csv"])
#============Data Framing and colum mapping===============
if csv_file is not None:
    try:
        df_raw = pd.read_csv(csv_file, encoding='utf-8')

    except Exception:
        st.error("Error reading CSV file. Please ensure it is a valid CSV format.")
        df_raw = pd.read_csv(csv_file, encoding='latin-1')# encoding changes
        st.success(f"CSV file read successfully with shape: {df_raw.shape}")
    with st.expander('Map the columns'):
        cols = ["<None>"] + list(df_raw.columns)
        map_student = st.selectbox('Student Name column:', cols, index=cols.index('student_name') if 'student_name' in df_raw.columns else 0)
        map_year = st.selectbox('Year column:', cols, index=cols.index('year') if 'year' in df_raw.columns else 0)
        map_location = st.selectbox('Location column:', cols, index=cols.index('location') if 'location' in df_raw.columns else 0)
        map_remarks = st.selectbox('Remarks column:', cols, index=cols.index('remarks') if 'remarks' in df_raw.columns else 0)
        map_tech_stack = st.selectbox('Tech Stack column:', cols, index=cols.index('tech_stack') if 'tech_stack' in df_raw.columns else 0)
        map_date = st.selectbox('Date column:', cols, index=cols.index('date') if 'date' in df_raw.columns else 0)
        map_label = st.selectbox('Student sentiment Label column:', cols, index=cols.index('label') if 'label' in df_raw.columns else 0)
        map_phone = st.selectbox('Phone column:', cols, index=cols.index('phone') if 'phone' in df_raw.columns else 0)
        map_callid = st.selectbox('Call ID column:', cols, index=cols.index('call_id') if 'call_id' in df_raw.columns else 0)
    def pick(colname):
        return df_raw[colname] if colname != "<None>" else  df_raw [colname]
    
    #data frame Network linking as per model training.
    df = pd.DataFrame({
        'Student_name': pick(map_student) if map_student != "<None>" else pd.Series([None])* len(df_raw),
        'Student_year': pick(map_year) if map_year != "<None>" else pd.Series([None])* len(df_raw),
        'Student_location': pick(map_location) if map_location != "<None>" else pd.Series([None])* len(df_raw),
        'Student_remarks': pick(map_remarks) if map_remarks != "<None>" else pd.Series([None])* len(df_raw),
        'Student_tech_stack': pick(map_tech_stack) if map_tech_stack != "<None>" else pd.Series([None])* len(df_raw),
        'Student_date': pick(map_date) if map_date != "<None>" else pd.Series([None])* len(df_raw),
        'Student_label': pick(map_label) if map_label != "<None>" else pd.Series([None])* len(df_raw),
        'Student_phone': pick(map_phone) if map_phone != "<None>" else pd.Series([None])* len(df_raw),
        'Student_call_id': pick(map_callid) if map_callid != "<None>" else pd.Series([None])* len(df_raw),
    })
    if 'date' in df.columns:
            df['date_parsed'] = df['date'].apply(safe_parse_date)
    else:
        df['date_parsed'] = None
else:
     df = None
#====================Transcribe Process=========================
#mp3 files (audio files) -> text data.
transcripts=[]
if audio_files:
    st.subheader("step 2: Transcribe Audio")
    #  Hanlding the Transcribe process as per the model selection
    if asr_engine == "whisper":
        try:
            whisper_model = load_whisper(whisper_size)
        except Exception as e:
            st.error(str(e))
            whisper_model = None
    else:
        #vosk model 
        try:
            vosk_model = load_vosk(vosk_model_dir)
        except Exception as e:
            st.error(str(e))
            vosk_model = None

        # steps for the first time model will be downlaod and stored in the 
        # cache memory. till that time your screen will be freezed or not.
        # so there should be some mechanism to load the progress. or No.
        # Progress bar must be used to share the progress how much model is loaded.
        # 0% --> 20% --> 30% ---> 100%
    prog = st.progress(0)
    for i,f in enumerate(audio_files): #generator functions.
        audio_bytes = f.read()
        if asr_engine == "whisper" and 'whisper_model' in locals() and whisper_model is not None:
            text = transcribe_with_whisper(audio_bytes, whisper_model, f.name)
        elif asr_engine == "vosk" and 'vosk_model' in locals() and vosk_model is not None:
            text = transcribe_with_vosk(audio_bytes,vosk_model,f.name)
        else:
            text = '[asr not available]'
        transcripts.append({
            'call_id':os.path.splitext(os.path.basename(f.name))[0],
            'transcript_text' : text
        })
        prog.progress(int(((i+1)/len(audio_files))*100))
        #st.success(f"Transcribed {len(transcripts)} file(s)")
#make the new data frame using transcribe process.
if transcripts:
    df_tr = pd.DataFrame(transcripts)
else:
    df_tr = pd.DataFrame(columns=['call_id','transcripts_text'])#empty.


#merge the transcribe process with csv: data  set audio and text files.
#final date set with merging call logs and data set
if df is not None:
    st.subheader("3) Merge Logs + Transcripts")
    # Try join on call_id if available, else outer merge on index
    if "call_id" in df.columns and df["call_id"].notna().any() and not df_tr.empty:
        merged = pd.merge(df, df_tr, on="call_id", how="outer")
    else:
        # append transcripts as new rows if missing call_id
        merged = df.copy()
        if not df_tr.empty:
            extra = pd.DataFrame({
                "call_id": df_tr["call_id"],
                "student_name": None,
                "year": None,
                "tech_stack": None,
                "location": None,
                "remarks": "",
                "date": None,
                "label": None,
                "date_parsed": None,
                "transcript_text": df_tr["transcript_text"]
            })
            merged = pd.concat([merged, extra], ignore_index=True)

        merged["remarks"] = merged.get("remarks", pd.Series([""] * len(merged))).fillna("")
    if "transcript_text" not in merged.columns:
        merged["transcript_text"] = ""
    else:
        merged["transcript_text"] = merged["transcript_text"].fillna("")
        merged["transcript_text"] = (merged["remarks"].astype(str) + " " + merged["transcript_text"].astype(str)).str.strip()
        #merged["combined-text"] = merged["col1"]+ " " + merged["col2"]


    st.dataframe(merged.head(50), use_container_width=True)
else:
    merged = None

#Ml pipeline and skit Learn.
#model training and sentiment Analysis.
if merged is not None and len(merged)>0:
    st.subheader('step 4: sentiment Analysis')
    use_pretrained = False
    can_train = ("label" in merged.columns) and merged['label'].notna().any() and not use_pretrained
    print(can_train) 

    if can_train:
        st.write('Training steps IF-IDF + Logististic Regression on Provided labels.')
        with st.spinner('Traing Model...Please wait.'):
            try:
                Vectorizer, clf, report = train_sklearn_sentiment(merged["combined-text"].fillna(""), merged["label"])
                st.text("classification report (hold out test): \n".format(report))
                #Prediction Algorithms for the custom model.
                #x data : train
                #y data : test
                xall = Vectorizer.transform(merged["combined-text"].fillna(""))
                merged['sentiment'] = clf.predict(xall)
                merged['sentiment_score'] = np.nan
                model_used = 'custom_sklearn'
            except Exception as e:
                st.error(f'Training Failed:{e} using Pretrained model pipeline.')
                can_train = False
    #if training fail.
    if not can_train:
        with st.spinner('Runing pre_trained  sentiment model'):
            nlp = load_hf_pipeline()
            preds = []
            score = []
            for txt in merged["transcript_text"].fillna(""):
                print(merged.columns)

                try:
                    #limit  = 4096 #1024,2098,4096
                    r = nlp(txt[:4096])[0]
                    label = r['label'].lower()


                    #positive ,Negative, Neutral.
                    if label == 'positive':
                        preds.append('positive')
                    elif label == 'negative':
                        preds.append('negative')
                    elif label =='neutral':
                        preds.append('neutral')
                    else:
                        preds.append(label)
                        score.append(float(r.get('score',np.nan)))
                except Exception as e:
                    print(f"Exception can not train{str(e)}")
                    st.error(f"Exception can not train{str(e)}")
                    preds.append('neutral')
                    score.append(np.nan)
                merged['sentiment'] = preds
                merged['sentiment_score'] = score
                model_used = 'hf_distilbert'
    st.success(f'sentiment computed using (model_used)')
    st.balloons()
    # Analytics.
    #------------------------------------------------
    #location ,Tech_stack...
    #------------------------------------------------
    #locations,tech_stack, date ,year.
    st.subheader('5.Analytics')
    colA,colB,colC,colD,colE = st.columns(5)

    with colA:
        fig = px.pie(merged, names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

#location Analytics 
    with colB:
        if "location" in merged.columns:
            fig2 = px.bar(merged.fillna({"location": "Unknown"}), x="location", color="sentiment", title="Sentiment by Location")
            st.plotly_chart(fig2, use_container_width=True)

        # if merged is not None and 'location' in merged.columns:
        #     fig2 = px.bar(merged.fillna({'location':'unknown'}),
        #     x='location',color='sentiment',title='sentiment by location')
        #     st.plotly_chart(fig2,use_container_width=True)
    #tech stack Analytics
    with colC:
        if "tech_stack" in merged.columns:
            fig3 = px.bar(merged.fillna({"tech_stack": "Unknown"}), x="tech_stack", color="sentiment", title="Sentiment by Tech Stack")
            st.plotly_chart(fig3, use_container_width=True)

        #     st.plotly_chart(fig3,use_container_width=True)
        # if merged is not None and 'tech_stack' in merged.columns:
        #     fig3 = px.bar(merged.fillna({'tech_stack':'unknown'}),
        #     x='tech_stack',color='sentiment',title='sentiment by tech stack')
        #     st.plotly_chart(fig3,use_container_width=True)
    with colD:
        if "date" in merged.columns:
            fig4 = px.bar(merged.fillna({"date": "Unknown"}), 
            x="tech_stack", color="sentiment", title="Sentiment by date")
            st.plotly_chart(fig4, use_container_width=True)

    with colE:
        if "Year" in merged.columns:
            fig5 = px.bar(merged.fillna({"Year": "Unknown"}), x="year", color="sentiment", title="Sentiment by Tech Stack")
            st.plotly_chart(fig5, use_container_width=True)

                        
    #trends.
    if 'date_parsed' in merged.columns and merged['date_parsed'].notna().any():
        temp = merged.copy()
        temp['month'] = temp['date_parsed'].dt.to_period('M').astype('str')
        ts = temp.groupby(['month','sentiment']).size().reset_index(name='count')
        fig4 = px.line(ts,x='month',y='count',color='sentiment',markers=True,title='Monthely Sentiment Trends')
        st.plotly_chart(fig4,use_container_width=True)

    #Negative.keywords.
    st.markdown("### Top Negative Keywords")
    neg = merged[merged['sentiment'] == 'negative']['combined_text'].dropna()
    if len(neg)>=3:
        vec = TfidfVectorizer(max_features=60,stop_words='english')
        x = vec.fit_transform(neg)

        #summation.of Vector.
        sums = np.asarray(x.sum(axis=0)).ravel()
        #vocabarly
        vocab = np.array(vec.get_feature_names_out())
        kw_df = pd.DataFrame({
            'keyword':vocab,
            'score':sums

        }).sort_values('score',ascending=False).head(20)
        fig5 = px.bar(kw_df, y='score',title = 'Top Negative Keywords (TF-IDF Vector)')
        st.plotly_chart(fig5,use_container_width=True)
    else:
        st.info('Not Negative sample to extract keywords')
        
    #Recomandation.
    def gen_recos(df_:pd.DataFrame):
        recos = []
        #overall negativity
        total = len(df_)
        negc = (df_['sentiment']=='negative').sum()
        if total>0 and negc/total > 0.4:
            recos.append(f"Overall negative sentiment is high(>40%),hence consider intermediate coaching.{negc/total:.0.35%}")

        #positive :Locatiion and tech_stack
        if 'location' in df_.columns:
            df_.groupby('location')['sentiment'].apply(lambda s:(s == 'negative').mean()).sort_values(ascending=False)
            for loc , ratio in by_loc.items():
                if pd.notna(loc) and ratio >=0.35:
                   recos.append(f"Consider addressing negative sentiment in {loc}:Negative ratio {ratio:.0.35%}, Trail:reduce fees , Add evening batchs,")
        #Tech_stack...
        if 'tech_stack' in df_.columns:
            by_stack = df_.groupby('tech_stack')['sentiment'].apply(lambda s: (s=='negative').mean()).sort_values(ascending=False)
            for stack, ratio in by_stack.items():
                if pd.notna(stack) and ratio >=0.35:
                    tips = {
                        'Java':'Emphasize job Outcome',
                        'Python':'Offer Installment plan',
                        'mern':'Highlight project-based learning',
                        'Django':'Promote full-stack capabilities',
                        'AI':'Incorporate more real-world examples',
                        'JavaScript':'Enhance front-end frameworks',
                        'Ruby':'Focus on community engagement',
                    }
                    #extra tips..
                    extra =""
                    k = str(stack).lower()
                    for key,val in tips.items():
                        if key in k:
                            extra = ";"+val
                            break
                    recos.append(f"Consider addressing negative sentiment in {stack}:Negative ratio {ratio:.0.35%},Address Objection {extra},")


        #keyword based recomandation:...
        text_all = " ".join(df_.get('combined_text',pd.Series(dtype=str)).dropna().astype(str).tolist()).lower()

        #cases.
        #we will try to cover all the cases.
        #feees
        if any(k in text_all for k in ['fees','fee','expensive','pricey','costly','cost','money','worth']):
            recos.append("Address pricing concerns by highlighting value and offering flexible payment options.and many fees related Objection -> try to discount offer, or seats or we have EMI or by Pay part payments")
        #timing issuse
        if any(k in text_all for k in ['duration','time','long','short','weeks','months']):
            recos.append("Address concerns about course duration and flexibility by offering personalized learning paths.")
        #location related issues.
        if any(k in text_all for k in ['location','city','place','far','distance','travel']):
            recos.append("Address location-based concerns by highlighting online/Hybrid learning options and local support.")
        #class & mentor..
        if any(k in text_all for k in ['class','classes','instructor','teacher','mentor','professor','faculty','mentor','consultant']):
            recos.append("Highlight the expertise of instructors and the benefits of small class sizes, advertise one to one mentoriship program,dout solving session, whatshapp or group connect available 24X7.")
        #job related issues.
        if any(k in text_all for k in ['job','placement','hiring','interview','recruitment','career','opportunity','resume','cv']):
            recos.append("Emphasize job placement support and career services in your marketing,resume/interview prepration,softskill,resume building.")


        #recomandation
        recomandation = gen_recos(merged) if merged is not None else []
        if recomandation:
            st.subheader("Top Recommendations")
            for r in recomandation:
                st.write(f"- {r}")
        else:
            st.info("No specific recommendations based on the current data.")    

            
    st.subheader('6.Top Recommandation')
    #Process csv/Downloads CSv file.
    
    st.subheader("7.Export/Downloads File")
    if save_intermediate:
        out_csv = merged.copy()
        out_buf = io.StringIO()
        out_csv.to_csv(out_buf,index = False)
        st.download_button("Download Processed CSV",data = out_buf.getvalue),
    file_name = f"Softpro_Processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime = "text/csv" 
else:
    st.info("Upload at least a Csv or audio to proceed.")





#Button.
st.map(color='purple',size = 20)
st.button('Explore More Analysis')
   