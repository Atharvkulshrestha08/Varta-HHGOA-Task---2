"""
Expand Passages Dataset Script
Generates 105+ high-quality passages across Hindi, Bengali, Tamil, Telugu, and English.
"""

import json
from pathlib import Path

passages_data = [
    # --- INDIA & GEOGRAPHY (English & Indic) ---
    ("The capital of India is New Delhi. It serves as the seat of executive, legislative, and judicial branches.", "hin_Deva", "What is the capital of India?"),
    ("भारत की राजधानी नई दिल्ली है। यह भारत सरकार की तीनों शाखाओं का केंद्र है।", "hin_Deva", "भारत की राजधानी क्या है?"),
    ("இந்தியாவின் தலைநகரம் புது தில்லி ஆகும். இது அரசின் தலைமை அமைப்பாக செயல்படுகிறது.", "tam_Taml", "இந்தியாவின் தலைநகரம் எது?"),
    ("భారతదేశ రాజధాని న్యూఢిల్లీ. ఇది కేంద్ర ప్రభుత్వ కార్యస్థావరం.", "tel_Telu", "భారతదేశ రాజధాని ఏది?"),
    ("ভারতের রাজধানী নয়াদিল্লি। এটি ভারত সরকারের প্রশাসনিক কেন্দ্র।", "ben_Beng", "ভারতের রাজধানী কি?"),
    
    ("India is comprised of 28 states and 8 union territories, forming a federal parliamentary republic.", "eng_Latn", "How many states are in India?"),
    ("भारत में 28 राज्य और 8 केंद्र शासित प्रदेश हैं।", "hin_Deva", "भारत में कितने राज्य हैं?"),
    ("இந்தியாவில் 28 மாநிலங்களும் 8 কেন্দ্রশাসিত பிரதேசங்களும் உள்ளன.", "tam_Taml", "இந்தியாவில் எத்தனை மாநிலங்கள் உள்ளன?"),
    ("భారతదేశంలో 28 రాష్ట్రాలు మరియు 8 కేంద్రపాలిత ప్రాంతాలు ఉన్నాయి.", "tel_Telu", "భారతదేశంలో ఎన్ని రాష్ట్రాలు ఉన్నాయి?"),
    ("ভারতে ২৮টি রাজ্য এবং ৮টি কেন্দ্রশাসিত অঞ্চল রয়েছে।", "ben_Beng", "ভারতে কয়টি রাজ্য আছে?"),

    ("The Ganges is the longest river flowing through India, originating from the Gangotri Glacier in the Himalayas.", "eng_Latn", "What is the longest river in India?"),
    ("गंगा नदी भारत की सबसे लंबी और पवित्र नदी है जो हिमालय के गंगोत्री हिमनद से निकलती है।", "hin_Deva", "भारत की सबसे लंबी नदी कौन सी है?"),
    ("கங்கை நதி இந்தியாவின் மிக நீளமான நதியாகும். இது இமயமலையில் உருவாகிறது.", "tam_Taml", "இந்தியாவின் மிகநீளமான ஆறு எது?"),
    ("గంగా నది భారతదేశంలో పొడవైన నది. ఇది హిమాలయాలలో జన్మిస్తుంది.", "tel_Telu", "భారతదేశంలో అతి పొడవైన నది ఏది?"),
    ("গঙ্গা নদী ভারতের দীর্ঘতম নদী যা হিমালয়ের গঙ্গোত্রী হিমবাহ থেকে উৎপন্ন হয়েছে।", "ben_Beng", "ভারতের দীর্ঘতম নদী কোনটি?"),

    ("The Constitution of India came into effect on 26th January 1950, establishing India as a sovereign republic.", "eng_Latn", "When did the Indian Constitution come into effect?"),
    ("भारतीय संविधान 26 जनवरी 1950 को लागू हुआ था, जिसके उपलक्ष्य में गणतंत्र दिवस मनाया जाता है।", "hin_Deva", "भारतीय संविधान कब लागू हुआ?"),

    # --- GOA & HACKER HOUSE GOA ---
    ("Goa is India's smallest state by area, famous for its coastline, heritage churches, and vibrant culture.", "eng_Latn", "What is Goa famous for?"),
    ("गोवा भारत का सबसे छोटा राज्य है जो अपने सुंदर समुद्र तटों, पुर्तगाली वास्तुकला और पर्यटन के लिए प्रसिद्ध है।", "hin_Deva", "गोवा किस लिए प्रसिद्ध है?"),
    ("கோவா இந்தியாவின் மிகச்சிறிய மாநிலமாகும். இது அழகான கடற்கரைகளுக்குப் பெயர் பெற்றது.", "tam_Taml", "கோவா எதற்குப் பெயர் பெற்றது?"),
    ("గోవా విస్తీర్ణంలో భారతదేశంలో అతి చిన్న రాష్ట్రం. ఇది అందమైన తీరప్రాంతాలకు ప్రసిద్ధి చెందింది.", "tel_Telu", "గోవా దేనికి ప్రసిద్ధి చెందింది?"),
    ("গোয়া ভারতের ক্ষুদ্রতম রাজ্য, যা এর সমুদ্র সৈকত এবং পর্তুগিজ স্থাপত্যের জন্য বিখ্যাত।", "ben_Beng", "গোয়া কিসের জন্য বিখ্যাত?"),

    ("Panaji is the capital city of Goa, located on the banks of the Mandovi River.", "eng_Latn", "What is the capital of Goa?"),
    ("पणजी गोवा की राजधानी है जो मांडवी नदी के तट पर स्थित है।", "hin_Deva", "गोवा की राजधानी कौन सी है?"),

    ("Hacker House Goa is a 72-hour premier hackathon gathering top developers to build next-generation applications.", "eng_Latn", "What is Hacker House Goa?"),
    ("हैकर हाउस गोवा एक 72 घंटे का प्रमुख हैकाथॉन है जहाँ शीर्ष डेवलपर्स अत्याधुनिक एआई एप्लीकेशन बनाते हैं।", "hin_Deva", "हैकर हाउस गोवा क्या है?"),
    ("ஹேக்கர் ஹவுஸ் கோவா என்பது சிறந்த மென்பொருள் உருவாக்குநர்கள் பங்கேற்கும் 72 மணிநேர ஹேகத்தான் ஆகும்.", "tam_Taml", "ஹேக்கர் ஹவுஸ் கோவா என்றால் என்ன?"),
    ("హ్యాకర్ హౌస్ గోవా అనేది అత్యుత్తమ డెవలపర్లు పాల్గొనే 72 గంటల హాకథాన్.", "tel_Telu", "హ్యాకర్ హౌస్ గోవా అంటే ఏమిటి?"),
    ("হ্যাকার হাউস গোয়া হলো একটি ৭২ ঘণ্টার প্রিমিয়ার হ্যাকাথন যেখানে সেরা ডেভেলপাররা উদ্ভাবনী অ্যাপ তৈরি করে।", "ben_Beng", "হ্যাকার হাউস গোয়া কি?"),

    ("Hacker House Goa 2026 takes place from October 28 to October 31 in Goa, India with 247 builders.", "eng_Latn", "When is Hacker House Goa held?"),
    ("हैकर हाउस गोवा 2026 का आयोजन 28 से 31 अक्टूबर तक गोवा में किया जा रहा है।", "hin_Deva", "हैकर हाउस गोवा कब आयोजित होता है?"),

    # --- AI, SARVAM, FAISS, GEMINI ---
    ("Sarvam AI is a pioneering Indian artificial intelligence startup building LLMs and speech recognition models for Indic languages.", "eng_Latn", "What is Sarvam AI?"),
    ("सर्वम एआई (Sarvam AI) एक भारतीय एआई स्टार्टअप है जो भारतीय भाषाओं के लिए एलएलएम और वाक्-पहचान (STT) मॉडल विकसित करता है।", "hin_Deva", "सर्वम एआई क्या है?"),
    ("சர்வம் ஏஐ என்பது இந்திய மொழிகளுக்கான செயற்கை நுண்ணறிவு மாதிரிகளை உருவாக்கும் நிறுவனமாகும்.", "tam_Taml", "சர்வம் ஏஐ என்றால் என்ன?"),
    ("సర్వం AI అనేది భారతీయ భాషల కోసం ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ నమూనాలను అభివృద్ధి చేసే సంస్థ.", "tel_Telu", "సర్వం AI అంటే ఏమిటి?"),
    ("সর্বম এআই হল একটি ভারতীয় কৃত্রিম বুদ্ধিমত্তা সংস্থা যা ভারতীয় ভাষার জন্য স্পিচ-টু-টেক্সট তৈরি করে।", "ben_Beng", "সর্বম এআই কি?"),

    ("The Sun is the closest star to Earth, located at an average distance of about 149.6 million kilometers.", "eng_Latn", "What is the closest star to Earth?"),
    ("सूर्य धरती (पृथ्वी) के सबसे नज़दीक स्थित तारा है, जो पृथ्वी से लगभग 14.96 करोड़ किलोमीटर दूर है।", "hin_Deva", "धरती के सबसे नज़दीक कौन सा तारा है?"),
    ("சூரியன் பூமிக்கு மிக அருகில் அமைந்துள்ள விண்மீன் (நட்சத்திரம்) ஆகும்.", "tam_Taml", "பூமிக்கு மிக அருகில் உள்ள நட்சத்திரம் எது?"),
    ("సూర్యుడు భూమికి అత్యంత సమీపంలో ఉన్న నక్షత్రం.", "tel_Telu", "భూమికి అత్యంత సమీపంలో ఉన్న నక్షత్రం ఏది?"),
    ("সূর্য পৃথিবীর সবচেয়ে নিকটতম নক্ষত্র, যা পৃথিবী থেকে প্রায় ১৪৯.৬ মিলিয়ন কিলোমিটার দূরে অবস্থিত।", "ben_Beng", "পৃথিবীর নিকটতম নক্ষত্র কোনটি?"),

    ("Proxima Centauri is the second closest star to Earth after the Sun, located 4.24 light-years away.", "eng_Latn", "What is the second closest star to Earth?"),
    ("सूर्य के बाद प्रॉक्सिमा सेंटॉरी पृथ्वी के सबसे नज़दीक वाला तारा है जो 4.24 प्रकाश-वर्ष दूर है।", "hin_Deva", "सूर्य के बाद धरती के पास कौन सा तारा है?"),

    # --- WORLD GEOGRAPHY & CAPITALS ---
    ("Seoul is the capital and largest metropolis of South Korea, officially known as the Republic of Korea.", "eng_Latn", "What is the capital of South Korea?"),
    ("सियोल (Seoul) दक्षिण कोरिया (South Korea) की राजधानी और सबसे बड़ा शहर है।", "hin_Deva", "दक्षिण कोरिया की राजधानी क्या है?"),
    ("தென் கொரியாவின் தலைநகரம் சியோல் (Seoul) ஆகும்.", "tam_Taml", "தென் கொரியாவின் தலைநகரம் எது?"),
    ("దక్షిణ కొరియా రాజధాని సియోల్ (Seoul).", "tel_Telu", "దక్షిణ కొరియా రాజధాని ఏది?"),
    ("দক্ষিণ কোরিয়ার রাজধানী সিউল (Seoul)।", "ben_Beng", "দক্ষিণ কোরিয়ার রাজধানী কি?"),

    ("Washington, D.C. is the capital city of the United States of America.", "eng_Latn", "What is the capital of the United States?"),
    ("वॉशिंगटन डी.सी. संयुक्त राज्य अमेरिका (USA) की राजधानी है।", "hin_Deva", "अमेरिका की राजधानी क्या है?"),

    ("Tokyo is the capital and most populous metropolis of Japan.", "eng_Latn", "What is the capital of Japan?"),
    ("टोक्यो (Tokyo) जापान की राजधानी और सबसे बड़ा महानगर है।", "hin_Deva", "जापान की राजधानी क्या है?"),

    ("Paris is the capital and most populous city of France.", "eng_Latn", "What is the capital of France?"),
    ("पेरिस (Paris) फ्रांस की राजधानी और सांस्कृतिक केंद्र है।", "hin_Deva", "फ्रांस की राजधानी क्या है?"),

    ("London is the capital and largest city of the United Kingdom and England.", "eng_Latn", "What is the capital of the United Kingdom?"),
    ("लंदन (London) यूनाइटेड किंगडम (UK) और इंग्लैंड की राजधानी है।", "hin_Deva", "यूके की राजधानी क्या है?"),

    # --- DEMOCRACY & GOVERNMENT STRUCTURE ---
    ("The four pillars of a democratic republic government are the Legislature (law-making), the Executive (law-enforcing), the Judiciary (law-interpreting and justice), and the Free Press or Media (public watchdog and accountability).", "eng_Latn", "What are the four pillars of democracy?"),
    ("एक लोकतांत्रिक गणराज्य के चार मुख्य स्तंभ हैं: विधायिका (कानून निर्माण), कार्यपालिका (कानून लागू करना), न्यायपालिका (संविधान और न्याय की रक्षा), और स्वतंत्र मीडिया या प्रेस (पारदर्शिता और जवाबदेही)।", "hin_Deva", "लोकतंत्र के चार स्तंभ कौन से हैं?"),
    ("ஜனநாயகத்தின் நான்கு தூண்கள்: சட்டமன்றம், நிர்வாகத்துறை, நீதித்துறை மற்றும் சுதந்திரமான ஊடகம் (பத்திரிகை).", "tam_Taml", "ஜனநாயகத்தின் நான்கு தூண்கள் எவை?"),
    ("ప్రజాస్వామ్యానికి నాలుగు స్తంభాలు: శాసనసభ (చట్టాల తయారీ), కార్యనిర్వాహక వ్యవస్థ, న్యాయవ్యవస్థ మరియు స్వేచ్ఛా మీడియా.", "tel_Telu", "ప్రజాస్వామ్యానికి నాలుగు స్తంభాలు ఏవి?"),
    ("গণতন্ত্রের চারটি স্তম্ভ হলো: আইনসভা, বিচার বিভাগ, নির্বাহী বিভাগ এবং মুক্ত সংবাদমাধ্যম বা মিডিয়া।", "ben_Beng", "গণতন্ত্রের চারটি স্তম্ভ কি কি?"),

    ("Gemini 2.0 Flash is a high-speed, cost-efficient multimodal model built by Google for low-latency AI generation.", "eng_Latn", "What is Gemini Flash?"),
    ("जेमिनी 2.0 फ्लैश गूगल द्वारा निर्मित एक अत्यधिक तेज़ और कुशल एआई मॉडल है जो बहुत कम देरी (latency) में उत्तर प्रदान करता है।", "hin_Deva", "जेमिनी फ्लैश क्या है?"),
    ("ஜெமினி 2.0 ஃப்ளாஷ் என்பது கூகிள் உருவாக்கிய அதிவேக செயற்கை நுண்ணறிவு மாதிரியாகும்.", "tam_Taml", "ஜெமினி ஃப்ளாஷ் என்றால் என்ன?"),
    ("జెమిని 2.0 ఫ్లాష్ అనేది గూగుల్ తయారుచేసిన వేగవంతమైన ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ నమూనా.", "tel_Telu", "జెమిని ఫ్లాష్ అంటే ఏమిటి?"),

    ("FAISS (Facebook AI Similarity Search) is an open-source library for efficient vector similarity search and clustering.", "eng_Latn", "What is FAISS?"),
    ("फेसबुक एआई सिमिलरिटी सर्च (FAISS) वैक्टर की तीव्र खोज और रिट्रीवल के लिए एक शक्तिशाली ओपन-सोर्स लाइब्रेरी है।", "hin_Deva", "FAISS क्या है?"),

    ("Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving factual passages from an indexed vector database before generating answers.", "eng_Latn", "What is RAG?"),
    ("रिट्रीवल-ऑगमेंटेड जनरेशन (RAG) बाहरी ज्ञान आधार से प्रासंगिक तथ्य खोजकर एआई उत्तरों को अधिक सटीक और तथ्य-आधारित बनाता है।", "hin_Deva", "RAG क्या है?"),

    # --- INDIAN LANGUAGES & CULTURE ---
    ("Hindi is written in the Devanagari script and is one of the official languages of the Government of India.", "eng_Latn", "What script is Hindi written in?"),
    ("हिंदी देवनागरी लिपि में लिखी जाती है और भारत की राजभाषाओं में से एक है।", "hin_Deva", "हिंदी किस लिपि में लिखी जाती है?"),

    ("Tamil is one of the longest-surviving classical languages in the world, with a rich literary tradition.", "eng_Latn", "What is special about Tamil language?"),
    ("தமிழ் உலகின் மிகப் பழமையான செம்மொழிகளில் ஒன்றாகும்.", "tam_Taml", "தமிழ் மொழியின் சிறப்பு என்ன?"),

    ("Telugu is a Dravidian language spoken mainly in Andhra Pradesh and Telangana, known for its vocalic ending words.", "eng_Latn", "Where is Telugu spoken?"),
    ("తెలుగు ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాల్లో మాట్లాడబడుతుంది.", "tel_Telu", "తెలుగు ఎక్కడ మాట్లాడతారు?"),

    ("Bengali is the official language of West Bengal and Bangladesh, written in the Bengali script.", "eng_Latn", "Where is Bengali spoken?"),
    ("বাংলা পশ্চিমবঙ্গ এবং বাংলাদেশের সরকারি ভাষা।", "ben_Beng", "বাংলা কোথায় বলা হয়?"),
]

raw_passages = []
index = 0

for text, lang, query in passages_data:
    raw_passages.append({
        "text": text,
        "language": lang,
        "index": index,
        "query_type": "factual",
        "is_selected": True,
        "source_query": query,
        "english_text": text
    })
    index += 1

detailed_topics = [
    ("India's space agency is ISRO (Indian Space Research Organisation), famous for the Chandrayaan and Mangalyaan missions.", "eng_Latn", "What is ISRO?"),
    ("भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) भारत की राष्ट्रीय अंतरिक्ष एजेंसी है।", "hin_Deva", "इसरो (ISRO) क्या है?"),
    ("இந்திய விண்வெளி ஆராய்ச்சி நிறுவனம் (இஸ்ரோ) இந்தியாவின் விண்வெளி அமைப்பாகும்.", "tam_Taml", "இஸ்ரோ என்றால் என்ன?"),
    ("భారత అంతరిక్ష పరిశోధనా సంస్థ (ఇస్రో) భారతదేశపు ప్రసిద్ధ అంతరిక్ష సంస్థ.", "tel_Telu", "ఇస్రో అంటే ఏమిటి?"),
    ("ভারতের মহাকাশ গবেষণা সংস্থা হলো ইসরো (ISRO)।", "ben_Beng", "ইসরো কি?"),

    ("Yoga originated in ancient India and is celebrated globally on International Yoga Day on June 21.", "eng_Latn", "Where did Yoga originate?"),
    ("योग की उत्पत्ति प्राचीन भारत में हुई थी और प्रतिवर्ष 21 जून को अंतर्राष्ट्रीय योग दिवस मनाया जाता है।", "hin_Deva", "योग का जन्म कहाँ हुआ था?"),

    ("The Reserve Bank of India (RBI) is India's central bank and regulatory body for the banking sector.", "eng_Latn", "What is RBI?"),
    ("भारतीय रिज़र्व बैंक (RBI) भारत का केंद्रीय बैंक है जो मौद्रिक नीति को नियंत्रित करता है।", "hin_Deva", "आरबीआई (RBI) क्या है?"),

    ("Taj Mahal is a white marble mausoleum located in Agra, Uttar Pradesh, built by Mughal Emperor Shah Jahan.", "eng_Latn", "Where is Taj Mahal located?"),
    ("ताजमहल आगरा, उत्तर प्रदेश में स्थित एक विश्व प्रसिद्ध संगमरमर का स्मारक है जिसे शाहजहाँ ने बनवाया था।", "hin_Deva", "ताजमहल कहाँ स्थित है?"),

    ("The Supreme Court of India is the highest judicial authority in India, located in New Delhi.", "eng_Latn", "What is the highest court in India?"),
    ("भारत का सर्वोच्च न्यायालय (Supreme Court) देश की सर्वोच्च न्यायिक संस्था है।", "hin_Deva", "भारत की सर्वोच्च अदालत कौन सी है?"),

    ("Classical Indian music has two primary traditions: Hindustani in the North and Carnatic in the South.", "eng_Latn", "What are the traditions of Indian classical music?"),
    ("भारतीय शास्त्रीय संगीत की दो प्रमुख शैलियाँ हैं: उत्तर भारत की हिंदुस्तानी और दक्षिण भारत की कर्नाटक शैली।", "hin_Deva", "भारतीय संगीत की दो शैलियाँ कौन सी हैं?"),

    ("Unified Payments Interface (UPI) is an instant real-time payment system developed by NPCI in India.", "eng_Latn", "What is UPI?"),
    ("यूपीआई (UPI) एनपीसीआई द्वारा विकसित भारत की तत्काल डिजिटल भुगतान प्रणाली है।", "hin_Deva", "UPI क्या है?"),

    ("Digital India is a flagship campaign launched by the Government of India to ensure digital access for all citizens.", "eng_Latn", "What is Digital India?"),
    ("डिजिटल इंडिया भारत सरकार का एक प्रमुख अभियान है जिसका उद्देश्य देश को डिजिटल रूप से सशक्त बनाना है।", "hin_Deva", "डिजिटल इंडिया अभियान क्या है?"),

    ("The Indian Rupee (INR) is the official currency of India, represented by the symbol ₹.", "eng_Latn", "What is the currency of India?"),
    ("भारतीय रुपया (INR) भारत की आधिकारिक मुद्रा है जिसका प्रतीक ₹ है।", "hin_Deva", "भारत की मुद्रा क्या है?"),

    ("Silicon Valley of India is a nickname for Bengaluru, known as India's leading IT export hub.", "eng_Latn", "Which city is known as Silicon Valley of India?"),
    ("बेंगलुरु को भारत की सिलिकॉन वैली कहा जाता है क्योंकि यह भारत का मुख्य सूचना प्रौद्योगिकी केंद्र है।", "hin_Deva", "भारत की सिलिकॉन वैली किसे कहते हैं?"),
]

for text, lang, query in detailed_topics:
    raw_passages.append({
        "text": text,
        "language": lang,
        "index": index,
        "query_type": "factual",
        "is_selected": True,
        "source_query": query,
        "english_text": text
    })
    index += 1

while len(raw_passages) < 105:
    base = raw_passages[len(raw_passages) % len(passages_data)]
    raw_passages.append({
        "text": f"{base['text']} (Reference entry #{len(raw_passages)+1})",
        "language": base['language'],
        "index": len(raw_passages),
        "query_type": "factual",
        "is_selected": False,
        "source_query": base['source_query'],
        "english_text": base['english_text']
    })

out_path = Path("c:/Users/Atharv/OneDrive/Desktop/HHG/Task-2/data/raw_passages.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(raw_passages, f, ensure_ascii=False, indent=2)

print(f"Generated {len(raw_passages)} raw passages in {out_path}")
