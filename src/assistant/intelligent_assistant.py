# -*- coding: utf-8 -*-
# src/assistant/intelligent_assistant.py - UPDATED: Detailed diet plan, auto-email
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
import json
from datetime import datetime

@dataclass
class HealthAssistant:
    groq_client: object
    model: str
    lang: str = "en"
    user_context: dict = field(default_factory=dict)
    conversation_history: list = field(default_factory=list)
    consultation_stage: str = "gathering"
    questions_asked: int = 0
    
    # Store extracted health data
    health_summary: dict = field(default_factory=dict)

    SYSTEM_PROMPT_EN = """You are a friendly health assistant. Ask ONE simple question at a time.

RULES:
1. Ask ONLY ONE question per response
2. Keep responses SHORT (1-2 sentences)
3. Be friendly and conversational
4. NO paragraphs, NO multiple questions
5. After gathering info (5-6 questions), ask: "Is there anything else?"
6. When user says "no" or "nothing else", IMMEDIATELY provide recommendations

INFORMATION TO GATHER:
- Duration of symptoms
- Severity (1-10)
- Other symptoms
- Medications tried
- Age (for diet)
- Weight (for diet)
- Height (for diet)
- Activity level (sedentary/moderate/active)

EXAMPLE:
User: "I have fever"
You: "How long have you had the fever?"

User: "2 days"
You: "On a scale of 1-10, how severe is it?"

User: "7"
You: "Any other symptoms like headache or body aches?"

User: "Yes, headache"
You: "Have you taken any medication?"

User: "Paracetamol"
You: "What's your age?" (for diet calculation)

User: "25"
You: "What's your weight in kg?"

User: "65"
You: "What's your height in cm?"

User: "170"
You: "Are you sedentary, moderately active, or very active?"

User: "Moderate"
You: "Is there anything else you'd like to tell me?"

User: "No"
You: [PROVIDE FULL RECOMMENDATIONS IMMEDIATELY]"""

    SYSTEM_PROMPT_TE = """మీరు స్నేహపూర్వక ఆరోగ్య సహాయకుడు. ఒక్క సాధారణ ప్రశ్న మాత్రమే అడగండి.

నియమాలు:
1. ఒక్క ప్రశ్న మాత్రమే
2. చిన్న సమాధానాలు (1-2 వాక్యాలు)
3. స్నేహపూర్వకంగా
4. పేరాలు వద్దు
5. సమాచారం సేకరించిన తర్వాత అడగండి: "మరేదైనా చెప్పాలనుకుంటున్నారా?"
6. వినియోగదారు "లేదు" అన్నప్పుడు వెంటనే సిఫార్సులు అందించండి

సేకరించాలసిన సమాచారం:
- ఎంత కాలం
- తీవ్రత (1-10)
- ఇతర లక్షణాలు
- మందులు
- వయస్సు
- బరువు
- ఎత్తు
- చైతన్యం"""

    def _sys_prompt(self) -> str:
        base = self.SYSTEM_PROMPT_TE if self.lang == "te" else self.SYSTEM_PROMPT_EN
        safety = (
            "\n\nSAFETY RULES:\n"
            "- Never diagnose the user. If asked for a diagnosis, say you cannot diagnose and recommend a medical professional.\n"
            "- Do not provide exact prescription dosages or medical treatment plans.\n"
            "- Always keep responses short and ask only one simple question at a time while gathering information.\n"
            "- Use friendly, conversational language.\n"
            "- If the user shares a mood or says they feel happy/sad, acknowledge it and politely ask if they have any health concerns or symptoms.\n"
            "- When giving recommendations, clearly state this is educational guidance and not a diagnosis.\n"
            "- Do not make definitive statements such as 'you have' or 'you will'.\n"
        )
        return base + safety

    def _chat(self, messages: List[Dict[str, str]], max_tokens: int = 150, expect_question: bool = False) -> str:
        """Call Groq API and enforce response guardrails"""
        try:
            resp = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            raw_response = resp.choices[0].message.content.strip()
            return self._validate_assistant_response(raw_response, expect_question=expect_question)
        except Exception as e:
            print(f"Chat API error: {e}")
            return "I'm having trouble processing. Please try again."

    def _validate_assistant_response(self, response: str, expect_question: bool = False) -> str:
        """Ensure the model response follows guardrails."""
        text = (response or "").strip()
        if not text:
            if expect_question:
                return "Can you tell me more about your symptoms in one short sentence?"
            return "I'm sorry, I couldn't generate a response. Please try again."

        normalized = re.sub(r"\s+", " ", text).strip()
        lower = normalized.lower()

        if expect_question:
            # Force a question when gathering information.
            if "?" not in normalized or normalized.count("?") > 1:
                return "Can you tell me more about your symptoms in one short question?"
            if len(normalized.splitlines()) > 4 or len(normalized) > 220:
                return "Please answer briefly and tell me one symptom or concern."

        # Safety guard: disallow diagnostic certainty or prescription claims.
        if re.search(r"\b(i can diagnose|i cannot diagnose|i can't diagnose|you have|you will|definitely|certainly|surely)\b", lower):
            if expect_question:
                return "I cannot diagnose your condition. Can you describe any additional symptoms?"

        # Add a strong disclaimer for recommendation responses.
        if not expect_question and not re.search(r"\b(medical diagnosis|educational guidance|not a diagnosis|consult .* doctor|consult .* medical professional)\b", lower):
            normalized += "\n\nNote: This is educational guidance, not a medical diagnosis. Consult a healthcare professional for serious symptoms."

        return normalized

    def _is_mood_only_text(self, text: str) -> bool:
        """Detect non-health mood/status messages and avoid treating them as symptoms."""
        normalized = (text or "").strip().lower()
        if not normalized:
            return False

        mood_phrases = [
            "i am happy", "i'm happy", "i feel happy", "feeling happy",
            "i am good", "i'm good", "i feel good", "feeling good",
            "i am okay", "i'm okay", "i feel okay", "feeling okay",
            "i am fine", "i'm fine", "i feel fine", "feeling fine",
            "i am well", "i'm well", "i feel well", "feeling well",
            "i am sad", "i'm sad", "i feel sad", "feeling sad",
            "i am great", "i'm great", "feeling great"
        ]
        health_keywords = [
            "pain", "fever", "cough", "headache", "nausea", "vomit", "dizzy",
            "symptom", "hurt", "ache", "infection", "cold", "flu", "injury",
            "anxiety", "depression", "stress", "sore", "stomach", "diarrhea",
            "blood", "rash", "allergy", "breath", "chest", "vomiting"
        ]
        health_phrases = [
            "no cough", "no fever", "no pain", "not feeling well", "not well"
        ]

        if any(phrase in normalized for phrase in mood_phrases):
            if not any(keyword in normalized for keyword in health_keywords) and not any(phrase in normalized for phrase in health_phrases):
                return True
        return False

    def _extract_health_data(self) -> Dict:
        """Extract health data with proper error handling"""
        conversation_text = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in self.conversation_history[-20:]
        ])
        
        extraction_prompt = f"""Extract health info as JSON:

{conversation_text}

Return only JSON:
{{
    "chief_complaint": "main problem",
    "symptoms": ["symptom1", "symptom2"],
    "severity": "mild/moderate/severe",
    "duration": "2 days",
    "age": 30,
    "weight": 70,
    "height": 170,
    "sex": "male/female",
    "activity_level": "sedentary/moderate/active",
    "medications_tried": ["med1"],
    "condition_category": "fever/respiratory/digestive/pain/skin/other"
}}"""

        messages = [
            {"role": "system", "content": "Extract data. Return only JSON."},
            {"role": "user", "content": extraction_prompt}
        ]
        
        try:
            response = self._chat(messages, max_tokens=300)
            
            # Clean JSON
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
            
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                response = response[start:end]
            
            data = json.loads(response)
            
            # Ensure all fields exist with defaults
            defaults = {
                "chief_complaint": "health concern",
                "symptoms": ["symptom discussed"],
                "severity": "moderate",
                "duration": "recent",
                "age": 30,
                "weight": 70,
                "height": 170,
                "sex": "male",
                "activity_level": "moderate",
                "medications_tried": [],
                "condition_category": "other"
            }
            
            for key, default_val in defaults.items():
                if key not in data or data[key] is None:
                    data[key] = default_val
            
            # Ensure numeric fields are actually numbers
            for num_field in ["age", "weight", "height"]:
                try:
                    data[num_field] = int(data[num_field]) if data[num_field] else defaults[num_field]
                except (ValueError, TypeError):
                    data[num_field] = defaults[num_field]
            
            return data
            
        except Exception as e:
            print(f"Extraction error: {e}")
            # Return safe defaults
            return {
                "chief_complaint": "health concern",
                "symptoms": ["symptoms discussed"],
                "severity": "moderate",
                "duration": "recent",
                "age": 30,
                "weight": 70,
                "height": 170,
                "sex": "male",
                "activity_level": "moderate",
                "medications_tried": [],
                "condition_category": "other"
            }

    def _generate_detailed_diet_plan(self, health_data: Dict) -> str:
        """Generate DETAILED diet plan in the exact format you specified"""
        
        try:
            age = int(health_data.get("age", 30) or 30)
            weight = int(health_data.get("weight", 70) or 70)
            height = int(health_data.get("height", 170) or 170)
            sex = str(health_data.get("sex", "male") or "male").lower()
            activity = str(health_data.get("activity_level", "moderate") or "moderate").lower()
            condition = health_data.get("condition_category", "other")
            chief_complaint = health_data.get("chief_complaint", "health concern")
            
            # Calculate BMR (Mifflin-St Jeor)
            if sex == "female":
                bmr = 10 * weight + 6.25 * height - 5 * age - 161
            else:
                bmr = 10 * weight + 6.25 * height - 5 * age + 5
            
            # Activity multipliers
            multipliers = {"sedentary": 1.2, "moderate": 1.55, "active": 1.725}
            calories = int(bmr * multipliers.get(activity, 1.55))
            
            # Adjust for illness (reduce by 10-20%)
            if condition in ["fever", "digestive"]:
                calories = int(calories * 0.85)
            
            protein = int(weight * 1.0)  # 1g per kg
            carbs = int(calories * 0.55 / 4)  # 55% of calories
            fat = int(calories * 0.25 / 9)  # 25% of calories
            
        except (ValueError, TypeError) as e:
            print(f"BMR calculation error: {e}")
            calories = 1800
            protein = 70
            carbs = 250
            fat = 40
        
        if self.lang == "te":
            diet_plan = f"""
## 🍽️ వ్యక్తిగత ఆహార ప్రణాళిక

### రోజువారీ పోషక లక్ష్యాలు:
- **కేలరీలు:** {calories} kcal
- **ప్రోటీన్:** {protein} g
- **కార్బోహైడ్రేట్లు:** {carbs} g
- **కొవ్వు:** {fat} g

### ఉదయం అల్పాహారం - తేలికపాటి & పోషకమైనది ({int(calories * 0.20)} kcal)
**ఉత్తమ సమయం:** ఉదయం 8:00
  • గంజి: 1.5 కప్పులు → జీర్ణం చేయడం సులభం, హైడ్రేషన్
  • పెరుగు: 1 గ్లాసు → చల్లగా, ప్రోబయోటిక్స్

### మధ్యాహ్నం భోజనం - సమతుల్యత & తేలిక ({int(calories * 0.35)} kcal)
**ఉత్తమ సమయం:** మధ్యాహ్నం 1:00
  • అన్నం: 1 కప్పు → శక్తి మూలం
  • పప్పు (పెసలు): 1 కప్పు → సులభంగా జీర్ణమయ్యే ప్రోటీన్
  • ఆవిరిలో వండిన కూరగాయలు: 1/2 కప్పు → విటమిన్లు, ఫైబర్

### రాత్రి భోజనం - తేలికైనది & త్వరగా ({int(calories * 0.25)} kcal)
**ఉత్తమ సమయం:** సాయంత్రం 7:00
  • రోటీ: 2 చిన్నవి → ఫైబర్, పోషణ
  • తేలికపాటి కూరగాయల కూర: 1 కప్పు → బరువు లేకుండా పోషకాలు

### హైడ్రేషన్
  • లక్ష్యం: రోజుకు 3 లీటర్ల నీరు
  • ప్రతి 1-2 గంటలకు ద్రవాలు తాగండి. చల్లని పానీయాలు తీసుకోవద్దు.

### సప్లిమెంట్స్ (వైద్యుడు సిఫార్సు చేస్తే)
  • విటమిన్ C (వైద్యుడు సిఫార్సు చేస్తే)
  • జింక్ (రోగనిరోధక శక్తి కోసం)

### ఈ ఆహారం ఎందుకు:
{chief_complaint} కోసం సులభంగా జీర్ణమయ్యే ఆహారాలు, తగినంత హైడ్రేషన్ మరియు రోగనిరోధక-మద్దతు పోషకాలను ప్రాధాన్యత ఇస్తుంది.

### తినకూడని ఆహారాలు:
  • మసాలా మరియు నూనె ఆహారాలు తప్పించండి
  • వేపుడు వంటకాలు తప్పించండి
  • పెరుగు/మజ్జిగ తప్ప పాల ఉత్పత్తులు తప్పించండి
  • చల్లని లేదా రిఫ్రిజిరేటెడ్ ఆహారాలు తప్పించండి
  • గొంతు నొప్పి ఉంటే సిట్రస్ తినవద్దు

### పర్యవేక్షించాల్సినవి:
ఆకలి మెరుగుదల, శక్తి స్థాయిలు మరియు సాధారణ భాగాలు తినగలగడం పర్యవేక్షించండి. లక్షణాలు మెరుగుపడినప్పుడు క్రమంగా సాధారణ ఆహారానికి తిరిగి రండి.
"""
        else:
            diet_plan = f"""
## 🍽️ Personalized Diet Plan

### DAILY NUTRITIONAL TARGETS:
- **Calories:** {calories} kcal
- **Protein:** {protein} g
- **Carbohydrates:** {carbs} g
- **Fat:** {fat} g

### BREAKFAST - Light & Nourishing ({int(calories * 0.20)} kcal)
**Best Time:** 8:00 AM
  • Rice porridge (ganji): 1.5 cups
    → Easy to digest, hydrating
  • Buttermilk: 1 glass
    → Cooling, probiotics

### LUNCH - Balanced & Light ({int(calories * 0.35)} kcal)
**Best Time:** 1:00 PM
  • Rice: 1 cup
    → Energy source
  • Moong dal: 1 cup
    → Easy-to-digest protein
  • Steamed vegetables: 1/2 cup
    → Vitamins, fiber

### DINNER - Light & Early ({int(calories * 0.25)} kcal)
**Best Time:** 7:00 PM
  • Chapati: 2 small
    → Fiber, sustenance
  • Light vegetable curry: 1 cup
    → Nutrients without heaviness

### HYDRATION
  • Target: 3 liters of water daily
  • Drink fluids every 1-2 hours. Avoid cold drinks.

### SUPPLEMENTS (if recommended by doctor)
  • Vitamin C (if doctor recommends)
  • Zinc (for immunity)

### WHY THIS DIET:
This diet prioritizes easy-to-digest foods, adequate hydration, and immune-supporting nutrients while being gentle on the digestive system during {chief_complaint}.

### FOODS TO AVOID:
  • Avoid spicy and oily foods
  • No fried items
  • Skip dairy except curd/buttermilk
  • Avoid cold or refrigerated foods
  • No citrus if throat is sore

### WHAT TO MONITOR:
Track appetite improvement, energy levels, and ability to eat normal portions. Gradually return to regular diet as symptoms improve.
"""
        
        return diet_plan

    def _generate_recommendations(self, health_data: Dict) -> str:
        """Generate complete recommendations with DETAILED diet plan"""
        
        condition = health_data.get("condition_category", "other")
        symptoms = health_data.get("symptoms", [])
        severity = health_data.get("severity", "moderate")
        chief_complaint = health_data.get("chief_complaint", "health concern")
        duration = health_data.get("duration", "recent")
        
        if self.lang == "te":
            prompt = f"""ఈ పరిస్థితికి పూర్తి సిఫార్సులు అందించండి:

పరిస్థితి: {chief_complaint}
లక్షణాలు: {', '.join(symptoms)}
తీవ్రత: {severity}
కాలం: {duration}

ఈ ఆకృతిలో అందించండి:

## 🥼 ఆరోగ్య సిఫార్సులు

### సహజ పరిష్కారాలు (ఇంటి చికిత్స)
- **పరిష్కారం 1 ({condition} కోసం నిర్దిష్టం):** ఎలా చేయాలి మరియు ఎందుకు సహాయపడుతుంది
- **పరిష్కారం 2 ({condition} కోసం నిర్దిష్టం):** ఎలా చేయాలి మరియు ఎందుకు సహాయపడుతుంది
- **పరిష్కారం 3 ({condition} కోసం నిర్దిష్టం):** ఎలా చేయాలి మరియు ఎందుకు సహాయపడుతుంది

### వైద్య సలహా
- **మందు 1:** {condition} కోసం నిర్దిష్ట మోతాదు మరియు ఉద్దేశ్యం
- **మందు 2:** {condition} కోసం నిర్దిష్ట మోతాదు మరియు ఉద్దేశ్యం

### స్వీయ-సంరక్షణ చిట్కాలు
{condition} కోసం నిర్దిష్ట చిట్కాలు

### హెచ్చరిక సంకేతాలు - వైద్యుడిని చూడండి ఒకవేళ:
- {condition} కోసం నిర్దిష్ట హెచ్చరిక సంకేతం 1
- {condition} కోసం నిర్దిష్ట హెచ్చరిక సంకేతం 2
- {condition} కోసం నిర్దిష్ట హెచ్చరిక సంకేతం 3

అన్నీ {condition} కు అత్యంత నిర్దిష్టంగా చేయండి."""
        else:
            prompt = f"""Provide complete recommendations for this condition:

Condition: {chief_complaint}
Symptoms: {', '.join(symptoms)}
Severity: {severity}
Duration: {duration}

Format:

## 🥼 Health Recommendations

### Natural Remedies (Home Care)
- **Remedy 1 (specific to {condition}):** How to do and why it helps
- **Remedy 2 (specific to {condition}):** How to do and why it helps
- **Remedy 3 (specific to {condition}):** How to do and why it helps

### Medical Advice
- **Medicine 1:** Specific dosage and purpose for {condition}
- **Medicine 2:** Specific dosage and purpose for {condition}

### Self-Care Tips
Specific tips for {condition}

### Warning Signs - See Doctor If:
- Specific warning sign 1 for {condition}
- Specific warning sign 2 for {condition}
- Specific warning sign 3 for {condition}

Make everything HIGHLY SPECIFIC to {condition}."""

        messages = [
            {"role": "system", "content": "You are a medical expert. Provide disease-specific recommendations."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            recommendations = self._chat(messages, max_tokens=1500)
            
            # Generate detailed diet plan
            diet_plan = self._generate_detailed_diet_plan(health_data)
            
            # Combine recommendations + diet plan
            full_response = f"""{recommendations}

---

{diet_plan}"""
            
            return full_response
        except Exception as e:
            print(f"Recommendation generation error: {e}")
            return "Unable to generate recommendations. Please consult a healthcare professional."

    def process_message(self, text: str) -> str:
        """Process message - ONE question at a time, auto-generate when done"""
        self.conversation_history.append({"role": "user", "content": text})
        self.questions_asked += 1

        if self._is_mood_only_text(text):
            response = (
                "I'm glad you're feeling happy! "
                "If you'd like to discuss your health, I'm here to help anytime."
            )
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        self.questions_asked += 1
        
        # Check if user is done
        user_is_done = self._detect_completion_keywords(text)
        
        if user_is_done:
            # User said "no" - AUTOMATICALLY generate recommendations
            try:
                health_data = self._extract_health_data()
                self.health_summary = health_data  # Store for email
                recommendations = self._generate_recommendations(health_data)
                response = recommendations
            except Exception as e:
                print(f"Error generating recommendations: {e}")
                if self.lang == "te":
                    response = "క్షమించండి, సిఫార్సులు రూపొందించడంలో సమస్య ఉంది. దయచేసి మళ్లీ ప్రయత్నించండి."
                else:
                    response = "Sorry, there was an error generating recommendations. Please try again."
        else:
            # Continue asking questions
            stage_instruction = f"\n\nYou have asked {self.questions_asked} questions. Ask ONE more simple question (1-2 sentences). After 6-7 questions, ask if there's anything else."
            
            messages = [
                {"role": "system", "content": self._sys_prompt() + stage_instruction}
            ]
            
            recent_history = self.conversation_history[-10:]
            messages.extend(recent_history)
            
            response = self._chat(messages, expect_question=True)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def generate_final_consultation_summary(self) -> Dict:
        """Generate final summary for email"""
        try:
            health_data = self._extract_health_data()
            self.health_summary = health_data
            
            recommendations = self._generate_recommendations(health_data)
            
            full_summary = f"""{recommendations}

───────────────────────────────────────────────────────

**📅 Generated:** {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p IST')}

**⚠️ DISCLAIMER:** This is educational guidance, not a medical diagnosis. Consult a healthcare professional for severe symptoms."""
            
            return {
                "recommendations": recommendations,
                "health_data": health_data,
                "full_summary": full_summary
            }
        except Exception as e:
            print(f"Summary generation error: {e}")
            return {
                "recommendations": "Error generating summary",
                "health_data": {},
                "full_summary": "Please consult a healthcare professional."
            }

    def _detect_completion_keywords(self, text: str) -> bool:
        """Detect if user is done"""
        text_lower = text.lower().strip()
        if not text_lower:
            return False

        exact_done = {
            "no", "nope", "nah", "done",
            "nothing", "nothing else", "nothing more",
            "that's all", "thats all", "that's it", "thats it",
            "అంతே", "మరేమీ లేదు"
        }

        if text_lower in exact_done:
            return True

        # Allow common completion phrases even when user adds punctuation or extra words.
        if re.search(r"\b(nothing else|nothing more|that's all|thats all|that's it|thats it|మరేమీ లేదు|అంతే)\b", text_lower):
            return True

        # Allow short affirmative completion responses like "no, that's all" but avoid matching
        # partial facts such as "I have no cough" or "no fever yet".
        if re.fullmatch(r"(no|nope|nah|done)[,\.\s]*(that's all|thats all|that's it|thats it|nothing else|nothing more)?", text_lower):
            return True

        return False

    def summarize_session(self, messages_from_db: List[Dict]) -> str:
        """Generate summary for email"""
        self.conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages_from_db if not m.get("is_internal", False)
        ]
        
        summary_dict = self.generate_final_consultation_summary()
        return summary_dict["full_summary"]