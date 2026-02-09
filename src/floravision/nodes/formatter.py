"""
FloraVision AI - Node 8: Response Formatting
=============================================

PURPOSE:
    Assembles the final user-facing response.
    Formats output in the mandatory markdown structure.
    Adds rescan suggestion if confidence is low.

POSITION IN PIPELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │  ... → Node 6 → Node 7 → [Node 8: Formatter] → END             │
    │                                ▲                                │
    │                                │                                │
    │                            YOU ARE HERE                         │
    │                          (FINAL NODE)                           │
    └─────────────────────────────────────────────────────────────────┘

READS FROM STATE:
    - All previous fields (this is the final assembly node)

WRITES TO STATE:
    - final_response (complete markdown output)
    - rescan_suggested (boolean)

CONNECTS TO:
    - Previous: nodes/safety.py (Node 7)
    - Next: None (returns to app.py for display)

OUTPUT FORMAT (as per spec):
    ## 🩺 Plant Diagnosis
    ## 📝 Care Plan
    ## 🚫 What Not To Do
    ## 🌤️ Seasonal Insight
    ## 💡 Pro Tip
"""

import json
from pathlib import Path
from ..state import PlantState
from .symptoms import get_symptom_display_name


# Load knowledge for display names
PLANTS_PATH = Path(__file__).parent.parent / "knowledge" / "plants.json"

with open(PLANTS_PATH) as f:
    PLANTS_DATA = json.load(f)


def formatter_node(state: PlantState) -> dict:
    """
    Node 8: Response Formatting
    
    Final node that assembles all information into a 
    beautifully formatted markdown response.
    
    Args:
        state: Complete PlantState with all fields populated
        
    Returns:
        dict with final_response and rescan_suggested
    """
    # Check if rescan should be suggested
    rescan_suggested = _should_suggest_rescan(state)
    
    # Build the response
    response = _build_response(state, rescan_suggested)
    
    return {
        "final_response": response,
        "rescan_suggested": rescan_suggested
    }


def _should_suggest_rescan(state: PlantState) -> bool:
    """
    Determine if we should suggest a rescan.
    
    Suggest rescan if:
    - Confidence is Low
    - Very few symptoms detected (might have missed some)
    - Conflicting symptoms detected
    """
    # Low confidence
    if state.confidence == "Low":
        return True
    
    # Very few detections with low individual confidence
    if len(state.yolo_detections) == 1:
        if state.yolo_detections[0].confidence < 0.6:
            return True
    
    return False


def _build_response(state: PlantState, rescan_suggested: bool) -> str:
    """
    Build the complete formatted response with rich, descriptive content.
    """
    sections = []
    
    plant_info = PLANTS_DATA.get(state.plant_name, PLANTS_DATA.get("unknown", {}))
    plant_display = _get_plant_display_name(state.plant_name)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Health Assessment (Doctor's Summary)
    # ═══════════════════════════════════════════════════════════════
    
    health_emoji = "🟢" if state.is_healthy else ("🟡" if state.severity == "Mild" else ("🟠" if state.severity == "Moderate" else "🔴"))
    health_status = _get_health_status_text(state)
    
    assessment = f"""## 🩺 Health Assessment

### Overall Status: {health_emoji} {health_status}

**Patient:** {plant_display}

{_get_doctor_summary(state, plant_info)}"""
    
    sections.append(assessment)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: Detailed Diagnosis
    # ═══════════════════════════════════════════════════════════════
    
    # Format detected symptoms with details
    if state.yolo_detections:
        symptom_details = []
        for d in state.yolo_detections:
            display_name = get_symptom_display_name(d.label)
            conf_pct = int(d.confidence * 100)
            symptom_details.append(f"  - **{display_name}** (confidence: {conf_pct}%)")
        symptoms_text = "\n".join(symptom_details)
        
        # Add category context
        if state.symptoms_grouped:
            categories = list(state.symptoms_grouped.keys())
            category_text = ", ".join(c.title() for c in categories)
            symptoms_text = f"*Stress Categories: {category_text}*\n\n{symptoms_text}"
    else:
        symptoms_text = "✅ No visible symptoms detected - your plant appears healthy!"
    
    severity_display = state.severity if state.severity else "None (Healthy)"
    confidence_explanation = _get_confidence_explanation(state.confidence)
    
    diagnosis = f"""## 🔬 Detailed Diagnosis

**Severity Level:** {severity_display}
**Diagnostic Confidence:** {state.confidence or 'Medium'} - {confidence_explanation}

### Detected Symptoms
{symptoms_text}"""
    
    # Add causes if present
    if state.causes:
        diagnosis += "\n\n### Likely Causes\n"
        for cause in state.causes:
            diagnosis += f"- {cause}\n"
    
    sections.append(diagnosis)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: About Your Plant
    # ═══════════════════════════════════════════════════════════════
    
    if state.plant_name != "unknown":
        plant_profile = f"""## 🌱 About Your {plant_display.split('(')[0].strip()}

**Scientific Name:** *{plant_info.get('scientific_name', 'Unknown')}*
**Light Needs:** {plant_info.get('light', 'Moderate indirect light')}
**Water Needs:** {plant_info.get('water_frequency', 'When top inch of soil is dry')}
**Common Issues:** {', '.join(plant_info.get('common_issues', ['None documented'])[:3])}
**Toxicity:** {plant_info.get('toxicity', 'Check before exposing to pets/children')}"""
        sections.append(plant_profile)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: Treatment Plan
    # ═══════════════════════════════════════════════════════════════
    
    care = "## 📋 Treatment Plan\n\n"
    
    care += "### 🚨 Immediate Actions\n"
    care += "*What to do in the next 24-48 hours:*\n\n"
    for i, action in enumerate(state.care_immediate, 1):
        care += f"{i}. {action}\n"
    
    care += "\n### 📅 Ongoing Care Schedule\n"
    care += "*Maintain these practices for best results:*\n\n"
    for action in state.care_ongoing:
        care += f"- {action}\n"
    
    sections.append(care.strip())
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: What Not To Do (Warnings)
    # ═══════════════════════════════════════════════════════════════
    
    dont = "## ⚠️ Common Mistakes to Avoid\n\n"
    dont += "*These actions can worsen your plant's condition:*\n\n"
    for item in state.dont_do:
        dont += f"❌ {item}\n"
    
    sections.append(dont.strip())
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: Seasonal Insight
    # ═══════════════════════════════════════════════════════════════
    
    season_emoji = {"spring": "🌸", "summer": "☀️", "autumn": "🍂", "winter": "❄️"}.get(state.season, "🌤️")
    seasonal = f"""## {season_emoji} Seasonal Care ({state.season.title()})

{state.seasonal_insight or 'Consider the current season when caring for your plant.'}"""
    sections.append(seasonal)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 7: Expert Tip
    # ═══════════════════════════════════════════════════════════════
    
    tip = f"""## 💡 Expert Tip

> {state.pro_tip or 'Every plant is unique - observe and learn from yours!'}"""
    sections.append(tip)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 8: Follow-Up Recommendation
    # ═══════════════════════════════════════════════════════════════
    
    followup = _get_followup_recommendation(state)
    sections.append(followup)
    
    # ═══════════════════════════════════════════════════════════════
    # OPTIONAL: Rescan Suggestion
    # ═══════════════════════════════════════════════════════════════
    if rescan_suggested:
        rescan = """> 📸 **For better accuracy**, try scanning the leaf closer under natural light."""
        sections.append(rescan)
    
    # Join all sections with dividers
    return "\n\n---\n\n".join(sections)


def _get_health_status_text(state: PlantState) -> str:
    """Get human-readable health status."""
    if state.is_healthy:
        return "Excellent Health"
    elif state.severity == "Mild":
        return "Minor Issues Detected"
    elif state.severity == "Moderate":
        return "Attention Needed"
    elif state.severity == "Critical":
        return "Urgent Care Required"
    return "Under Observation"


def _get_doctor_summary(state: PlantState, plant_info: dict) -> str:
    """Generate a doctor-style summary paragraph."""
    plant_name = state.plant_name.replace("_", " ").title()
    
    if state.is_healthy:
        return f"""Your {plant_name} is in **excellent condition**! The foliage appears vibrant, and no visible signs of disease, pests, or nutrient deficiencies were detected. This plant is thriving in its current environment.

**Prognosis:** Continue current care routine. Your plant is well-maintained. 🌿"""
    
    elif state.severity == "Mild":
        symptoms = list(state.symptoms_grouped.keys())
        symptom_text = " and ".join(s.title() for s in symptoms)
        return f"""Your {plant_name} is showing **early signs of {symptom_text} stress**. These symptoms are minor and easily treatable with prompt attention. The overall health of the plant remains stable.

**Prognosis:** Full recovery expected within 1-2 weeks with proper care. 💪"""
    
    elif state.severity == "Moderate":
        return f"""Your {plant_name} requires **attention**. Multiple stress indicators suggest the plant is struggling with its current conditions. Without intervention, the condition may deteriorate.

**Prognosis:** Recovery expected within 2-4 weeks with consistent treatment and environmental adjustments. ⚡"""
    
    elif state.severity == "Critical":
        return f"""Your {plant_name} is in **critical condition** and requires **immediate intervention**. Serious symptoms detected that could lead to plant loss if untreated. Act quickly but don't panic - many plants recover with proper care.

**Prognosis:** Guarded - recovery possible with aggressive treatment. Monitor daily. 🚨"""
    
    return f"Your {plant_name} is currently under observation. Follow the care recommendations below."


def _get_confidence_explanation(confidence: str) -> str:
    """Explain what the confidence level means."""
    explanations = {
        "High": "Multiple clear indicators support this diagnosis",
        "Medium": "Diagnosis based on visible symptoms with reasonable certainty",
        "Low": "Limited data available - consider rescanning for better accuracy"
    }
    return explanations.get(confidence, "Based on available visual data")


def _get_followup_recommendation(state: PlantState) -> str:
    """Get follow-up scanning recommendation."""
    if state.is_healthy:
        return """## 📆 Follow-Up

Your plant is healthy! **Recommended next scan:** 2-4 weeks, or if you notice any changes in leaf color, texture, or growth patterns."""
    
    elif state.severity == "Critical":
        return """## 📆 Follow-Up

⚠️ **Critical condition requires close monitoring.** 
- Scan again in **3-5 days** to track recovery
- Document any changes with photos
- If condition worsens, consider consulting a local nursery expert"""
    
    elif state.severity == "Moderate":
        return """## 📆 Follow-Up

**Recommended next scan:** 1 week after starting treatment to monitor progress. Look for improvement in leaf color and new growth."""
    
    else:
        return """## 📆 Follow-Up

**Recommended next scan:** 1-2 weeks to confirm improvement. Minor issues typically resolve quickly with proper care."""


def _get_plant_display_name(plant_name: str) -> str:
    """
    Get a nice display name for the plant.
    """
    if plant_name == "unknown":
        return "Unknown Plant (generic care provided)"
    
    plant_info = PLANTS_DATA.get(plant_name, {})
    scientific = plant_info.get("scientific_name", "")
    
    display = plant_name.replace("_", " ").title()
    if scientific:
        display += f" (*{scientific}*)"
    
    return display
