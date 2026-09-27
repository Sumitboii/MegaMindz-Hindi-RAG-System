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
SLV-III ने वर्ष 1980 में रोहिणी उपग्रह (Rohini Satellite) को सफलतापूर्वक पृथ्वी की कक्षा में स्थापित किया।

**Source (citation):**
```text
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.86
page: 8 · section: "6 · इसरो के वर्ष और SLV-III" · chunk_id: 14 · score: 0.79
page: 8 · section: "6 · इसरो के वर्ष और SLV-III" · chunk_id: 13 · score: 0.76
page: 17 · section: "15 · जीवन-रेखा (टाइमलाइन)" · chunk_id: 31 · score: 0.74
page: 17 · section: "15 · जीवन-रेखा (टाइमलाइन)" · chunk_id: 30 · score: 0.73
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
डॉ. ए.पी.जे. अब्दुल कलाम को वर्ष 1997 में भारत के सर्वोच्च नागरिक सम्मान 'भारत रत्न' से सम्मानित किया गया।

**Source (citation):**
```text
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 38 · score: 0.88
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.86
page: 16 · section: "14 · पुरस्काार एवं सम्माान" · chunk_id: 29 · score: 0.77
page: 20 · section: "18 · संस्थााओं एवं शब्दों की शब्दाावली" · chunk_id: 35 · score: 0.68
page: 4 · section: "2 · रामेश्वरम में बचपन" · chunk_id: 5 · score: 0.68
```

**Retrieval Evaluation:**
- Expected Source Pages: `[16, 17, 20, 22]`
- Retrieved Top-5 Pages: `[21, 21, 16, 20, 4]`
- Retrieval Status: **`PASS`**

---

## Query 3 (Hindi)
**Query:**
> पोखरण-II में कलाम की क्या भूमिका थी?

**Answer:**
पोखरण-II (1998 / ऑपरेशन शक्ति) परमाणु परीक्षणों में डॉ. कलाम ने मुख्य वैज्ञानिक सलाहकार एवं रक्षा अनुसंधान के समन्वयक के रूप में केंद्रीय भूमिका निभाई थी।

**Source (citation):**
```text
page: 10 · section: "8 · पोखरण-II: 1998 के परमाणु परीक्षण" · chunk_id: 17 · score: 0.89
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.81
page: 2 · section: "विषय-सूची" · chunk_id: 2 · score: 0.79
page: 12 · section: "10 · गुरु एवं शिक्षक" · chunk_id: 22 · score: 0.77
page: 12 · section: "10 · गुरु एवं शिक्षक" · chunk_id: 21 · score: 0.76
```

**Retrieval Evaluation:**
- Expected Source Pages: `[10, 17, 20]`
- Retrieved Top-5 Pages: `[10, 21, 2, 12, 12]`
- Retrieval Status: **`PASS`**

---

## Query 4 (English)
**Query:**
> Which institution did Kalam attend to study aeronautical engineering?

**Answer:**
Dr. Kalam attended Madras Institute of Technology (MIT), Chennai (मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी, चेन्नई) to study aeronautical engineering.

**Source (citation):**
```text
page: 6 · section: "4 · कॉलेज के वर्ष और उड़ाान का आकर्षण" · chunk_id: 9 · score: 0.61
page: 6 · section: "4 · कॉलेज के वर्ष और उड़ाान का आकर्षण" · chunk_id: 10 · score: 0.59
page: 20 · section: "18 · संस्थााओं एवं शब्दों की शब्दाावली" · chunk_id: 35 · score: 0.59
page: 9 · section: "7 · निर्देशित मिसाइल कार्यक्रम" · chunk_id: 16 · score: 0.58
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.58
```

**Retrieval Evaluation:**
- Expected Source Pages: `[6, 17, 20, 22]`
- Retrieved Top-5 Pages: `[6, 6, 20, 9, 21]`
- Retrieval Status: **`PASS`**

---

## Query 5 (English)
**Query:**
> Who co-wrote the autobiography, and in what year was it published?

**Answer:**
The autobiography 'Wings of Fire' (अग्नि की उड़ान) was co-written with Arun Tiwari (अरुण तिवारी) and published in the year 1999.

**Source (citation):**
```text
page: 1 · section: "एक अध्ययन एवं संदर्भ सहायिका" · chunk_id: 1 · score: 0.55
page: 3 · section: "1 · ए.पी.जे. अब्दुुल कलाम कौन थे?" · chunk_id: 4 · score: 0.55
page: 2 · section: "विषय-सूची" · chunk_id: 2 · score: 0.55
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.55
page: 3 · section: "1 · ए.पी.जे. अब्दुुल कलाम कौन थे?" · chunk_id: 3 · score: 0.55
```

**Retrieval Evaluation:**
- Expected Source Pages: `[1, 17, 22]`
- Retrieved Top-5 Pages: `[1, 3, 2, 21, 3]`
- Retrieval Status: **`PASS`**

---

## Query 6 (English)
**Query:**
> How and where did Kalam die in 2015?

**Answer:**
Dr. Kalam passed away on 27 July 2015 in Shillong, Meghalaya, while delivering a lecture to students at IIM Shillong.

**Source (citation):**
```text
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 38 · score: 0.83
page: 15 · section: "13 · अंतिम क्षण तक शिक्षक (2007-2015)" · chunk_id: 26 · score: 0.82
page: 22 · section: "20 · त्वरित तथ्य-सारांश" · chunk_id: 39 · score: 0.76
page: 22 · section: "20 · त्वरित तथ्य-सारांश" · chunk_id: 40 · score: 0.75
page: 21 · section: "19 · विचार एवं बोध-प्रश्न" · chunk_id: 37 · score: 0.74
```

**Retrieval Evaluation:**
- Expected Source Pages: `[15, 17, 22]`
- Retrieved Top-5 Pages: `[21, 15, 22, 22, 21]`
- Retrieval Status: **`PASS`**

---

