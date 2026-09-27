# Test Query Results & Retrieval Evaluation Log
**Target Document:** `Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf`
**Embedding Model:** `intfloat/multilingual-e5-base`
**Retriever:** Hybrid (Dense Cosine Similarity + BM25 Sparse Search)
**Total Document Chunks:** 40

---
## Query 1 (Hindi)
**Query:**
> SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?

**Answer:**
जुलाई 1980 में, SLV-III ने रोहिणी उपग्रह को सफलतापूर्वक कक्षा में स्थापित किया, जिससे भारत उन थोड़े-से देशों के समूह में आ गया जो अपने रॉकेटों से उपग्रह प्रक्षेपित कर सकते थे।

**Source (citation):**
```text
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.86
page: 8 · section: "6 · इसरो के वर्ष और SLV-III" · chunk_id: 14 · score: 0.79
page: 8 · section: "6 · इसरो के वर्ष और SLV-III" · chunk_id: 13 · score: 0.76
page: 17 · section: "15 · जीवन-रेखा (टाइमलाइन)" · chunk_id: 31 · score: 0.74
page: 17 · section: "15 · जीवन-रेखा (टाइमलाइन)" · chunk_id: 30 · score: 0.72
```

**Retrieval Evaluation:**
- Expected Source Pages: `[8, 17, 20, 22]`
- Retrieved Top-5 Pages: `[21, 8, 8, 17, 17]`
- Retrieval Status: **`PASS`**

---

## Query 2 (Hindi)
**Query:**
> कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?

**Answer:**
| भारत रत्न | भारत का सर्वोच्च नागरिक सम्मान, 1997 में कलाम को प्रदान। |

**Source (citation):**
```text
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 38 · score: 0.88
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.87
page: 16 · section: "14 · पुरस्कार एवं सम्मान" · chunk_id: 29 · score: 0.78
page: 8 · section: "6 · इसरो के वर्ष और SLV-III" · chunk_id: 13 · score: 0.69
page: 20 · section: "18 · संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 35 · score: 0.69
```

**Retrieval Evaluation:**
- Expected Source Pages: `[16, 17, 20, 22]`
- Retrieved Top-5 Pages: `[21, 21, 16, 8, 20]`
- Retrieval Status: **`PASS`**

---

## Query 3 (Hindi)
**Query:**
> पोखरण-II में कलाम की क्या भूमिका थी?

**Answer:**
कलाम, जो उस समय रक्षा मंत्री के वैज्ञानिक सलाहकार तथा डीआरडीओ के प्रमुख के रूप में सेवारत थे, ने परमाणु ऊर्जा विभाग के वैज्ञानिकों के साथ मिलकर एक केंद्रीय संगठनात्मक भूमिका निभाई। 8 · पोखरण-II: 1998 के परमाणु परीक्षण मई 1998 में, भारत ने राजस्थान के पोखरण में भूमिगत परमाणु परीक्षणों की एक शृंंखला की, जिसे पोखरण-II या ऑपरेशन शक्ति के नाम से जाना जाता है।

**Source (citation):**
```text
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.86
page: 10 · section: "8 · पोखरण-II: 1998 के परमाणु परीक्षण" · chunk_id: 17 · score: 0.86
page: 11 · section: "9 · अग्नि" · chunk_id: 20 · score: 0.79
page: 2 · section: "विषय-सूची" · chunk_id: 2 · score: 0.76
page: 12 · section: "10 · गुरु एवं शिक्षक" · chunk_id: 22 · score: 0.74
```

**Retrieval Evaluation:**
- Expected Source Pages: `[10, 17, 20]`
- Retrieved Top-5 Pages: `[21, 10, 11, 2, 12]`
- Retrieval Status: **`PASS`**

---

## Query 4 (English)
**Query:**
> Which institution did Kalam attend to study aeronautical engineering?

**Answer:**
उन्होंने वैमानिकी इंजीनियरिंग पढ़ने के लिए चेन्नई के मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी (एमआईटी) में प्रवेश लिया। जब उन्होंने वैमानिकी इंजीनियरिंग के बारे में जाना, तभी उन्हें अपना असली बुलावा मिला: वे उड़ने वाली मशीनें बनाना चाहते थे।

**Source (citation):**
```text
page: 6 · section: "4 · कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 9 · score: 0.61
page: 6 · section: "4 · कॉलेज के वर्ष और उड़ान का आकर्षण" · chunk_id: 10 · score: 0.59
page: 20 · section: "18 · संस्थाओं एवं शब्दों की शब्दावली" · chunk_id: 35 · score: 0.59
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.59
page: 2 · section: "विषय-सूची" · chunk_id: 2 · score: 0.58
```

**Retrieval Evaluation:**
- Expected Source Pages: `[6, 17, 20, 22]`
- Retrieved Top-5 Pages: `[6, 6, 20, 21, 2]`
- Retrieval Status: **`PASS`**

---

## Query 5 (English)
**Query:**
> Who co-wrote the autobiography, and in what year was it published?

**Answer:**
अग्नि की उड़ान, जो 1999 में प्रकाशित हुई और वैज्ञानिक अरुण तिवारी के साथ मिलकर लिखी गई, उनकी आत्मकथा है।

**Source (citation):**
```text
page: 3 · section: "1 · ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 4 · score: 0.55
page: 1 · section: "एक अध्ययन एवं संदर्भ सहायिका" · chunk_id: 1 · score: 0.55
page: 2 · section: "विषय-सूची" · chunk_id: 2 · score: 0.55
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.55
page: 3 · section: "1 · ए.पी.जे. अब्दुल कलाम कौन थे?" · chunk_id: 3 · score: 0.54
```

**Retrieval Evaluation:**
- Expected Source Pages: `[1, 17, 22]`
- Retrieved Top-5 Pages: `[3, 1, 2, 21, 3]`
- Retrieval Status: **`PASS`**

---

## Query 6 (English)
**Query:**
> How and where did Kalam die in 2015?

**Answer:**
27 जुलाई 2015 को, कलाम शिलांग के भारतीय प्रबंधन संस्थान में विद्यार्थियों को व्याख्यान देते समय गिर पड़े और कुछ ही देर बाद 83 वर्ष की आयु में उनका निधन हो गया।

**Source (citation):**
```text
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 38 · score: 0.83
page: 15 · section: "13 · अंतिम क्षण तक शिक्षक (2007-2015)" · chunk_id: 26 · score: 0.82
page: 22 · section: "20 · त्वरित तथ्य-सारांश" · chunk_id: 39 · score: 0.76
page: 22 · section: "20 · त्वरित तथ्य-सारांश" · chunk_id: 40 · score: 0.75
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.75
```

**Retrieval Evaluation:**
- Expected Source Pages: `[15, 17, 22]`
- Retrieved Top-5 Pages: `[21, 15, 22, 22, 21]`
- Retrieval Status: **`PASS`**

---

