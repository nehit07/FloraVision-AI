"""
FloraVision AI - PDF Report Generator
======================================

PURPOSE:
    Generates professional PDF reports from diagnosis data.
    Uses HTML/CSS template for beautiful formatting with
    FloraVision branding.

USAGE:
    from floravision.utils.pdf_report import generate_pdf_report
    pdf_bytes = generate_pdf_report(state)
"""

import io
from datetime import datetime
from xhtml2pdf import pisa

from ..state import PlantState
from ..nodes.symptoms import get_symptom_display_name


def generate_pdf_report(state: PlantState) -> bytes:
    """
    Generate a professional PDF report from the PlantState.
    
    Args:
        state: Complete PlantState with diagnosis data
        
    Returns:
        PDF file as bytes
    """
    html_content = _build_html_report(state)
    
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(
        io.StringIO(html_content),
        dest=pdf_buffer
    )
    
    if pisa_status.err:
        raise Exception(f"PDF generation failed with error: {pisa_status.err}")
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def _build_html_report(state: PlantState) -> str:
    """Build the HTML content for the report."""
    
    # Get plant display name
    plant_name = state.plant_name.replace("_", " ").title()
    if plant_name.lower() == "unknown":
        plant_name = "Unknown Plant"
    
    # Health status
    if state.is_healthy:
        health_status = "Excellent Health"
        health_color = "#22c55e"
        health_emoji = "🟢"
    elif state.severity == "Mild":
        health_status = "Minor Issues Detected"
        health_color = "#eab308"
        health_emoji = "🟡"
    elif state.severity == "Moderate":
        health_status = "Attention Needed"
        health_color = "#f97316"
        health_emoji = "🟠"
    elif state.severity == "Critical":
        health_status = "Urgent Care Required"
        health_color = "#ef4444"
        health_emoji = "🔴"
    else:
        health_status = "Under Observation"
        health_color = "#6b7280"
        health_emoji = "⚪"
    
    # Format symptoms
    if state.yolo_detections:
        symptoms_html = "<ul>"
        for d in state.yolo_detections:
            display_name = get_symptom_display_name(d.label)
            conf_pct = int(d.confidence * 100)
            symptoms_html += f'<li><strong>{display_name}</strong> ({conf_pct}% confidence)</li>'
        symptoms_html += "</ul>"
    else:
        symptoms_html = '<p class="healthy-text">✅ No visible symptoms detected - plant appears healthy!</p>'
    
    # Format causes
    if state.causes:
        causes_html = "<ul>"
        for cause in state.causes:
            causes_html += f"<li>{cause}</li>"
        causes_html += "</ul>"
    else:
        causes_html = "<p>No specific causes identified.</p>"
    
    # Format care plans
    immediate_html = "<ol>"
    for action in state.care_immediate:
        immediate_html += f"<li>{action}</li>"
    immediate_html += "</ol>"
    
    ongoing_html = "<ul>"
    for action in state.care_ongoing:
        ongoing_html += f"<li>{action}</li>"
    ongoing_html += "</ul>"
    
    # Format warnings
    warnings_html = "<ul class='warning-list'>"
    for item in state.dont_do:
        warnings_html += f"<li>❌ {item}</li>"
    warnings_html += "</ul>"
    
    # Season emoji
    season_emoji = {"spring": "🌸", "summer": "☀️", "autumn": "🍂", "winter": "❄️"}.get(state.season, "🌤️")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>FloraVision AI - Plant Health Report</title>
        <style>
            @page {{
                size: A4;
                margin: 1.5cm;
            }}
            
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #1f2937;
                background: white;
            }}
            
            .header-table {{
                width: 100%;
                border-bottom: 3px solid #22c55e;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }}
            
            .logo-icon {{
                font-size: 36pt;
                vertical-align: middle;
            }}
            
            .logo-title {{
                font-size: 22pt;
                color: #166534;
                font-weight: bold;
            }}
            
            .tagline {{
                color: #6b7280;
                font-size: 10pt;
            }}
            
            .report-meta {{
                text-align: right;
                font-size: 9pt;
                color: #6b7280;
            }}
            
            .status-banner {{
                padding: 12px 15px;
                border-radius: 8px;
                color: white;
                margin-bottom: 15px;
            }}
            
            .status-banner h2 {{
                font-size: 14pt;
                margin: 0;
            }}
            
            .status-banner .patient {{
                font-size: 11pt;
                margin: 3px 0 0 0;
            }}
            
            .section {{
                margin-bottom: 15px;
                page-break-inside: avoid;
            }}
            
            .section h3 {{
                font-size: 12pt;
                color: #166534;
                border-bottom: 1px solid #d1d5db;
                padding-bottom: 4px;
                margin-bottom: 8px;
            }}
            
            .section p {{
                margin-bottom: 6px;
            }}
            
            .section ul, .section ol {{
                margin-left: 20px;
                margin-bottom: 8px;
            }}
            
            .section li {{
                margin-bottom: 4px;
            }}
            
            .summary-box {{
                background: #f0fdf4;
                padding: 12px;
                border-left: 4px solid #22c55e;
                margin-bottom: 15px;
            }}
            
            .prognosis {{
                background: white;
                padding: 6px 10px;
                margin-top: 8px;
            }}
            
            .highlight-green {{ color: #166534; font-weight: bold; }}
            .highlight-yellow {{ color: #a16207; font-weight: bold; }}
            .highlight-orange {{ color: #c2410c; font-weight: bold; }}
            .highlight-red {{ color: #dc2626; font-weight: bold; }}
            
            .healthy-text {{
                color: #166534;
                font-weight: bold;
            }}
            
            .treatment-table {{
                width: 100%;
                margin-bottom: 15px;
            }}
            
            .treatment-table td {{
                width: 50%;
                vertical-align: top;
                padding: 8px;
            }}
            
            .immediate-box {{
                background: #fef3c7;
                border: 1px solid #f59e0b;
                padding: 10px;
            }}
            
            .ongoing-box {{
                background: #dbeafe;
                border: 1px solid #3b82f6;
                padding: 10px;
            }}
            
            .treatment-title {{
                font-size: 11pt;
                font-weight: bold;
                margin-bottom: 3px;
            }}
            
            .treatment-subtitle {{
                font-size: 9pt;
                color: #6b7280;
                margin-bottom: 6px;
            }}
            
            .warnings-box {{
                background: #fef2f2;
                padding: 10px;
                border-left: 4px solid #ef4444;
                margin-bottom: 15px;
            }}
            
            .warning-item {{
                color: #991b1b;
            }}
            
            .seasonal-box {{
                background: #f0f9ff;
                padding: 10px;
                margin-bottom: 15px;
            }}
            
            .tip-box {{
                background: #fefce8;
                padding: 10px 12px;
                border-left: 4px solid #eab308;
                font-style: italic;
                margin-bottom: 15px;
            }}
            
            .followup-box {{
                background: #f3f4f6;
                padding: 10px;
                margin-bottom: 15px;
            }}
            
            .footer {{
                margin-top: 20px;
                padding-top: 12px;
                border-top: 2px solid #22c55e;
                text-align: center;
                font-size: 9pt;
                color: #6b7280;
            }}
            
            .disclaimer {{
                font-size: 8pt;
                margin-top: 4px;
            }}
        </style>
    </head>
    <body>
        <!-- Header -->
        <table class="header-table">
            <tr>
                <td>
                    <span class="logo-icon">🌿</span>
                    <span class="logo-title">FloraVision AI</span><br/>
                    <span class="tagline">Plant Health Diagnosis Report</span>
                </td>
                <td class="report-meta">
                    <strong>Date:</strong> {timestamp}<br/>
                    <strong>Confidence:</strong> {state.confidence or 'Medium'}
                </td>
            </tr>
        </table>
        
        <!-- Health Status Banner -->
        <div class="status-banner" style="background-color: {health_color};">
            <h2>{health_emoji} {health_status}</h2>
            <p class="patient">Patient: {plant_name}</p>
        </div>
        
        <!-- Doctor's Summary -->
        <div class="summary-box">
            <h3>🩺 Diagnosis Summary</h3>
            {_get_doctor_summary_html(state, plant_name)}
        </div>
        
        <!-- Detected Symptoms -->
        <div class="section">
            <h3>🔬 Detected Symptoms</h3>
            {symptoms_html}
        </div>
        
        <!-- Likely Causes -->
        <div class="section">
            <h3>🔍 Likely Causes</h3>
            {causes_html}
        </div>
        
        <!-- Treatment Plan -->
        <div class="section">
            <h3>📋 Treatment Plan</h3>
            <table class="treatment-table">
                <tr>
                    <td>
                        <div class="immediate-box">
                            <p class="treatment-title">🚨 Immediate Actions</p>
                            <p class="treatment-subtitle">Do this in the next 24-48 hours</p>
                            {immediate_html}
                        </div>
                    </td>
                    <td>
                        <div class="ongoing-box">
                            <p class="treatment-title">📅 Ongoing Care</p>
                            <p class="treatment-subtitle">Maintain these practices</p>
                            {ongoing_html}
                        </div>
                    </td>
                </tr>
            </table>
        </div>
        
        <!-- What Not To Do -->
        <div class="warnings-box">
            <h3>⚠️ Common Mistakes to Avoid</h3>
            {warnings_html}
        </div>
        
        <!-- Seasonal Insight -->
        <div class="seasonal-box">
            <h3>{season_emoji} Seasonal Care ({state.season.title()})</h3>
            <p>{state.seasonal_insight or 'Consider current season when caring for your plant.'}</p>
        </div>
        
        <!-- Expert Tip -->
        <div class="tip-box">
            <h3>💡 Expert Tip</h3>
            <p>{state.pro_tip or 'Every plant is unique - observe and learn from yours!'}</p>
        </div>
        
        <!-- Follow-Up -->
        <div class="followup-box">
            <h3>📆 Follow-Up Recommendation</h3>
            <p>{_get_followup_text(state)}</p>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Generated by FloraVision AI</p>
            <p class="disclaimer">This report is for informational purposes only. For serious plant issues, consult a local nursery expert.</p>
        </div>
    </body>
    </html>
    """
    
    return html


def _get_doctor_summary_html(state: PlantState, plant_name: str) -> str:
    """Generate doctor-style summary HTML."""
    if state.is_healthy:
        return f"""
        <p>Your <strong>{plant_name}</strong> is in <span class="highlight-green">excellent condition</span>! 
        The foliage appears vibrant, and no visible signs of disease, pests, or nutrient deficiencies were detected. 
        This plant is thriving in its current environment.</p>
        <p class="prognosis"><strong>Prognosis:</strong> Continue current care routine. Your plant is well-maintained. 🌿</p>
        """
    
    elif state.severity == "Mild":
        symptoms = list(state.symptoms_grouped.keys()) if state.symptoms_grouped else ["general"]
        symptom_text = " and ".join(s.title() for s in symptoms)
        return f"""
        <p>Your <strong>{plant_name}</strong> is showing <span class="highlight-yellow">early signs of {symptom_text} stress</span>. 
        These symptoms are minor and easily treatable with prompt attention. The overall health of the plant remains stable.</p>
        <p class="prognosis"><strong>Prognosis:</strong> Full recovery expected within 1-2 weeks with proper care. 💪</p>
        """
    
    elif state.severity == "Moderate":
        return f"""
        <p>Your <strong>{plant_name}</strong> requires <span class="highlight-orange">attention</span>. 
        Multiple stress indicators suggest the plant is struggling with its current conditions. 
        Without intervention, the condition may deteriorate.</p>
        <p class="prognosis"><strong>Prognosis:</strong> Recovery expected within 2-4 weeks with consistent treatment. ⚡</p>
        """
    
    elif state.severity == "Critical":
        return f"""
        <p>Your <strong>{plant_name}</strong> is in <span class="highlight-red">critical condition</span> and requires 
        <strong>immediate intervention</strong>. Serious symptoms detected that could lead to plant loss if untreated. 
        Act quickly but don't panic - many plants recover with proper care.</p>
        <p class="prognosis"><strong>Prognosis:</strong> Guarded - recovery possible with aggressive treatment. Monitor daily. 🚨</p>
        """
    
    return f"<p>Your {plant_name} is currently under observation. Follow the care recommendations below.</p>"


def _get_followup_text(state: PlantState) -> str:
    """Get follow-up recommendation text."""
    if state.is_healthy:
        return "Your plant is healthy! Recommended next scan: 2-4 weeks, or if you notice any changes in leaf color, texture, or growth patterns."
    elif state.severity == "Critical":
        return "⚠️ Critical condition requires close monitoring. Scan again in 3-5 days to track recovery. Document any changes with photos. If condition worsens, consult a local nursery expert."
    elif state.severity == "Moderate":
        return "Recommended next scan: 1 week after starting treatment to monitor progress. Look for improvement in leaf color and new growth."
    else:
        return "Recommended next scan: 1-2 weeks to confirm improvement. Minor issues typically resolve quickly with proper care."


def _get_css_styles() -> str:
    """Get CSS styles for the PDF report."""
    return """
    @page {
        size: A4;
        margin: 1.5cm;
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
        color: #1f2937;
        background: white;
    }
    
    /* Header */
    header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 15px;
        border-bottom: 3px solid #22c55e;
        margin-bottom: 20px;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .logo-icon {
        font-size: 40pt;
    }
    
    .logo-text h1 {
        font-size: 24pt;
        color: #166534;
        margin: 0;
    }
    
    .tagline {
        color: #6b7280;
        font-size: 10pt;
        margin: 0;
    }
    
    .report-meta {
        text-align: right;
        font-size: 9pt;
        color: #6b7280;
    }
    
    .report-meta p {
        margin: 2px 0;
    }
    
    /* Status Banner */
    .status-banner {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .status-emoji {
        font-size: 30pt;
    }
    
    .status-text h2 {
        font-size: 16pt;
        margin: 0;
    }
    
    .status-text .patient {
        font-size: 11pt;
        opacity: 0.9;
        margin: 3px 0 0 0;
    }
    
    /* Sections */
    .section {
        margin-bottom: 18px;
        page-break-inside: avoid;
    }
    
    .section h3 {
        font-size: 13pt;
        color: #166534;
        border-bottom: 1px solid #d1d5db;
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
    
    .section p {
        margin-bottom: 8px;
    }
    
    .section ul, .section ol {
        margin-left: 20px;
        margin-bottom: 10px;
    }
    
    .section li {
        margin-bottom: 5px;
    }
    
    /* Summary Section */
    .summary-section {
        background: #f0fdf4;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #22c55e;
    }
    
    .summary-section h3 {
        border-bottom: none;
    }
    
    .prognosis {
        background: white;
        padding: 8px 12px;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    .highlight-green { color: #166534; font-weight: bold; }
    .highlight-yellow { color: #a16207; font-weight: bold; }
    .highlight-orange { color: #c2410c; font-weight: bold; }
    .highlight-red { color: #dc2626; font-weight: bold; }
    
    .healthy-text {
        color: #166534;
        font-weight: 500;
    }
    
    /* Treatment Grid */
    .treatment-grid {
        display: flex;
        gap: 15px;
    }
    
    .treatment-box {
        flex: 1;
        padding: 12px;
        border-radius: 8px;
    }
    
    .treatment-box.immediate {
        background: #fef3c7;
        border: 1px solid #f59e0b;
    }
    
    .treatment-box.ongoing {
        background: #dbeafe;
        border: 1px solid #3b82f6;
    }
    
    .treatment-box h4 {
        font-size: 11pt;
        margin-bottom: 3px;
    }
    
    .treatment-box .subtitle {
        font-size: 9pt;
        color: #6b7280;
        margin-bottom: 8px;
    }
    
    /* Warnings */
    .warnings-section {
        background: #fef2f2;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
    }
    
    .warning-list li {
        color: #991b1b;
    }
    
    /* Seasonal */
    .seasonal-section {
        background: #f0f9ff;
        padding: 12px;
        border-radius: 8px;
    }
    
    /* Expert Tip */
    .tip-section blockquote {
        background: #fefce8;
        padding: 12px 15px;
        border-left: 4px solid #eab308;
        border-radius: 0 8px 8px 0;
        font-style: italic;
    }
    
    /* Follow-up */
    .followup-section {
        background: #f3f4f6;
        padding: 12px;
        border-radius: 8px;
    }
    
    /* Footer */
    footer {
        margin-top: 25px;
        padding-top: 15px;
        border-top: 2px solid #22c55e;
        text-align: center;
        font-size: 9pt;
        color: #6b7280;
    }
    
    .disclaimer {
        font-size: 8pt;
        margin-top: 5px;
    }
    """
