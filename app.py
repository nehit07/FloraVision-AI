"""
FloraVision AI - Streamlit Application
=======================================

PURPOSE:
    Main user interface for the plant health assistant.
    Provides camera capture and image upload functionality.

FEATURES:
    - 📷 Camera capture for live plant scanning
    - 📁 Image upload for existing photos
    - 🌡️ Season selection for context-aware advice
    - 📊 Beautiful diagnosis display with emoji sections

CONNECTIONS:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Streamlit App                              │
    │                          │                                      │
    │   Uses:                  │                                      │
    │   - floravision.graph    │  (run_diagnosis function)            │
    │   - floravision.state    │  (for type hints)                    │
    │                          │                                      │
    │   Displays:              │                                      │
    │   - final_response       │  (markdown from formatter node)      │
    └─────────────────────────────────────────────────────────────────┘

USAGE:
    streamlit run app.py
    OR
    uv run streamlit run app.py
"""

import streamlit as st
from datetime import datetime
import io
import os

# Import FloraVision components
from src.floravision.graph import run_diagnosis, run_diagnosis_full
from src.floravision.nodes.seasonal import get_season_from_month
from src.floravision.utils.pdf_report import generate_pdf_report, generate_batch_pdf_report
from src.floravision.utils.visuals import draw_detections
from src.floravision.utils.database import db_manager
from src.floravision.utils.chat_manager import chat_manager
from src.floravision.state import PlantState
import logging

# Get logger
logger = logging.getLogger("floravision.ui")

# ═══════════════════════════════════════════════════════════════════
# CACHING LAYER (PHASE 5)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_run_diagnosis(image_bytes: bytes, season: str, climate_zone: str, mock: bool):
    """Cached wrapper for the core diagnosis logic to reduce API costs."""
    logger.info(f"Cache miss for diagnosis (mock={mock})")
    return run_diagnosis_full(image_bytes, season, climate_zone, mock)


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FloraVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# UI THEME & STYLING (PHASE 4)
# ═══════════════════════════════════════════════════════════════════

# Premium Dark Theme CSS
bg_color = "#0e1117"
text_color = "#fafafa"
secondary_bg = "#262730"

st.markdown(f"""
<style>
    /* Global Background & Text */
    .stAppViewContainer {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stSidebar {{
        background-color: {secondary_bg};
    }}
    
    .main-header {{
        text-align: center;
        padding: 1rem 0;
        color: {text_color};
    }}
    .diagnosis-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }}
    .severity-banner {{
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .severity-critical {{ background-color: #ef4444; color: white; }}
    .severity-moderate {{ background-color: #f97316; color: white; }}
    .severity-mild {{ background-color: #eab308; color: black; }}
    .severity-healthy {{ background-color: #22c55e; color: white; }}
    .stImage {{
        border-radius: 1rem;
        transition: transform 0.3s ease;
    }}
    .stImage:hover {{
        transform: scale(1.02);
    }}
    div[data-testid="stMarkdownContainer"] h2 {{
        margin-top: 1.5rem;
        color: #22c55e;
    }}
    /* Custom button styling for Email Share */
    .share-btn {{
        display: block;
        width: 100%;
        height: 38px;
        line-height: 38px;
        text-align: center;
        text-decoration: none !important;
        border-radius: 4px;
        border: 1px solid #ff4b4b;
        background-color: transparent;
        color: #ff4b4b !important;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 500;
        margin-top: 10px;
    }}
    .share-btn:hover {{
        background-color: #ff4b4b;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/391/potted-plant_1fab4.png", width=80)
    st.title("FloraVision AI")
    st.markdown("*Your intelligent plant health assistant* 🌿")
    
    st.divider()
    
    # Season selector
    st.subheader("🌤️ Current Season")
    
    # Auto-detect season from current month
    current_month = datetime.now().month
    auto_season = get_season_from_month(current_month)
    
    season_options = ["spring", "summer", "autumn", "winter"]
    default_idx = season_options.index(auto_season)
    
    season = st.selectbox(
        "Select season for context-aware advice:",
        options=season_options,
        index=default_idx,
        format_func=lambda x: f"{x.title()} {'🌸' if x == 'spring' else '☀️' if x == 'summer' else '🍂' if x == 'autumn' else '❄️'}"
    )
    
    st.divider()
    
    # Climate selector (Phase 3)
    st.subheader("⚙️ Personalization")
    climate = st.selectbox(
        "Climate Zone",
        options=["Temperate", "Tropical", "Arid", "Arctic"],
        index=0,
        help="Helps refine advice for your specific environment"
    )
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    
    
    mock_mode = st.toggle(
        "Demo Mode",
        value=True,
        help="Uses simulated detections for demo. Turn off for real AI detection (requires API key)."
    )
    
    if not mock_mode:
        st.info("🔑 Make sure GOOGLE_API_KEY is set in your .env file")
    
    st.divider()
    
    # About section
    with st.expander("ℹ️ About FloraVision AI"):
        st.markdown("""
        **FloraVision AI** uses:
        - 🔍 **YOLOv8** for symptom detection
        - 🧠 **LangGraph** for reasoning pipeline
        - 🤖 **Gemini** for intelligent analysis
        
        Built with ❤️ for plant lovers everywhere.
        """)

# ═══════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.divider()
    page = st.radio(
        "Navigation",
        options=["🔬 New Diagnosis", "📊 History Dashboard"],
        index=0
    )

if page == "🔬 New Diagnosis":


    # ═══════════════════════════════════════════════════════════════════
    # MAIN CONTENT
    # ═══════════════════════════════════════════════════════════════════
    
    # Initialize session state for persistence (Phase 3 Fix)
    if "current_plant_state" not in st.session_state:
        st.session_state.current_plant_state = None
    if "current_image_bytes" not in st.session_state:
        st.session_state.current_image_bytes = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "is_batch" not in st.session_state:
        st.session_state.is_batch = False
    
    st.markdown("<h1 class='main-header'>🌿 FloraVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Upload or capture a plant image for instant health diagnosis</p>", unsafe_allow_html=True)
    
    # Input mode selector - prevents camera from opening unnecessarily
    input_mode = st.radio(
        "Choose input method:",
        options=["📁 Upload Image", "📷 Camera Capture"],
        horizontal=True,
        help="Select how you want to provide the plant image"
    )
    
    # Show only the selected input method
    camera_image = None
    uploaded_file = None
    
    if input_mode == "📷 Camera Capture":
        st.subheader("📷 Take a Photo")
        camera_image = st.camera_input("Point your camera at the plant")
    else:
        st.subheader("📁 Upload Images")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Upload clear photos of the plants you want to diagnose. You can select multiple files!"
        )
    
    st.divider()
    
    # Handle multiple files or single camera image
    all_images = []
    image_source = "None"
    if camera_image:
        all_images.append({"bytes": camera_image.getvalue(), "name": "Camera Capture"})
        image_source = "camera"
    elif uploaded_files:
        for f in uploaded_files:
            all_images.append({"bytes": f.getvalue(), "name": f.name})
        image_source = "upload"
    
    # Reset results if images have changed
    current_hashes = [hash(img["bytes"]) for img in all_images]
    if "last_image_hashes" not in st.session_state or current_hashes != st.session_state.last_image_hashes:
        st.session_state.current_plant_state = None
        st.session_state.batch_results = []
        st.session_state.messages = []
        st.session_state.last_image_hashes = current_hashes
    
    # Show image preview and analyze button
    if all_images:
        num_images = len(all_images)
        st.session_state.is_batch = num_images > 1
        
        if st.session_state.is_batch:
            st.info(f"📦 **Batch Mode Active:** {num_images} images selected.")
        
        col_preview, col_analyze = st.columns([1, 2])
        
        with col_preview:
            # Show first image or small gallery? Let's show first with a count.
            st.image(all_images[0]["bytes"], caption=f"{all_images[0]['name']} (and {num_images-1} others)" if num_images > 1 else all_images[0]["name"], width="stretch")
        
        with col_analyze:
            st.markdown("### Ready to Analyze! 🔬")
            st.markdown(f"""
            - **Source:** {'📷 Camera' if image_source == 'camera' else '📁 Upload'}
            - **Season:** {season.title()}
            - **Mode:** {'🎭 Demo' if mock_mode else '🤖 Live AI'}
            """)
            
            analyze_button = st.button(
                "🩺 Analyze Plant Health",
                type="primary",
                width="stretch"
            )
    
        if analyze_button:
            progress_text = "🔍 Analyzing your plant..." if not st.session_state.is_batch else f"📦 Analyzing {num_images} plants..."
            progress_bar = st.progress(0)
            
            with st.spinner(progress_text):
                try:
                    results = []
                    for i, img in enumerate(all_images):
                        if st.session_state.is_batch:
                            st.write(f"Processing {img['name']} ({i+1}/{num_images})...")
                        
                        plant_state = cached_run_diagnosis(
                            image_bytes=img["bytes"],
                            season=season,
                            climate_zone=climate,
                            mock=mock_mode
                        )
                        
                        # Store image reference for display
                        plant_state.image = img["bytes"]
                        results.append(plant_state)
                        
                        progress_bar.progress((i + 1) / num_images)
                        
                        # Save to database
                        try:
                            db_manager.save_diagnosis(plant_state)
                        except:
                            pass # Don't block processing if DB fails
                    
                    st.session_state.batch_results = results
                    st.session_state.current_plant_state = results[0] if results else None
                    st.session_state.messages = []
                    
                    st.success(f"✅ Analysis complete! {len(results)} plants processed.")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("Try enabling Demo Mode in the sidebar, or check your API key configuration.")

        # ═══════════════════════════════════════════════════════════════════
        # DISPLAY RESULTS (PERSISTENT VIA SESSION STATE)
        # ═══════════════════════════════════════════════════════════════════
        
        if st.session_state.batch_results:
            # Batch Summary Dashboard
            if st.session_state.is_batch:
                st.markdown("## 📊 Batch Summary")
                
                # Simple health stats for the batch
                total = len(st.session_state.batch_results)
                healthy = sum(1 for r in st.session_state.batch_results if r.is_healthy)
                issues = total - healthy
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Scanned", total)
                c2.metric("Healthy", healthy)
                c3.metric("Issues Detected", issues)
                
                # Batch Download Button
                with st.spinner("📄 Preparing full batch report..."):
                    try:
                        batch_pdf = generate_batch_pdf_report(st.session_state.batch_results)
                        st.download_button(
                            label="📥 Download Full Batch PDF Report",
                            data=batch_pdf,
                            file_name=f"floravision_batch_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            width="stretch"
                        )
                    except Exception as e:
                        st.error(f"Batch PDF error: {e}")
                
                # Plant Selection for detailed view
                plant_names = [f"Plant {i+1}: {r.plant_name.replace('_', ' ').title()}" for i, r in enumerate(st.session_state.batch_results)]
                selected_plant_idx = st.selectbox("View Detailed Report for:", range(len(plant_names)), format_func=lambda x: plant_names[x])
                plant_state = st.session_state.batch_results[selected_plant_idx]
                st.session_state.current_plant_state = plant_state # For chat context
            else:
                plant_state = st.session_state.batch_results[0]
            
            orig_image = plant_state.image
            
            # Annotated Image
            with st.expander("🖼️ View Annotated Detections", expanded=True):
                annotated_image = draw_detections(orig_image, plant_state.yolo_detections)
                st.image(annotated_image, caption="AI Detection Highlights", width="stretch")
            
            st.divider()
            
            # Severity Banner
            severity = plant_state.severity or ("Healthy" if plant_state.is_healthy else "Unknown")
            severity_class = f"severity-{severity.lower()}"
            st.markdown(f"""
                <div class="severity-banner {severity_class}">
                    Health Status: {severity.upper()}
                </div>
            """, unsafe_allow_html=True)
            
            # Display the diagnosis results in expanders
            st.markdown("# 🌿 Diagnosis Results")
            
            # Confidence Chart
            if plant_state.yolo_detections:
                with st.expander("📊 Detection Confidence", expanded=False):
                    labels = [d.label.replace('_', ' ').title() for d in plant_state.yolo_detections]
                    confs = [d.confidence for d in plant_state.yolo_detections]
                    st.bar_chart(dict(zip(labels, confs)))

            # Split final response into sections if possible or show in expander
            sections = plant_state.final_response.split("===SECTION_BREAK===")
            
            # Summary Expander
            with st.expander("🩺 Doctor's Summary", expanded=True):
                st.markdown(sections[0] if len(sections) > 0 else plant_state.final_response)
            
            # Detailed Diagnosis
            if len(sections) > 1:
                with st.expander("🔬 Detailed Analysis", expanded=False):
                    st.markdown(sections[1])
            
            # Treatment Plan
            treatment_idx = -1
            for i, s in enumerate(sections):
                if "Treatment Plan" in s or "Care Plan" in s:
                    treatment_idx = i
                    break
            
            if treatment_idx != -1:
                with st.expander("📋 Treatment Plan", expanded=True):
                    st.markdown(sections[treatment_idx])
            
            # Other sections in one more expander
            with st.expander("ℹ️ Additional Insights & Tips", expanded=False):
                remaining_sections = []
                for i, s in enumerate(sections):
                    if i not in [0, 1, treatment_idx] and i < len(sections):
                        remaining_sections.append(s)
                st.markdown("\n\n---\n\n".join(remaining_sections))

            # Generate PDF and Sharing
            st.divider()
            
            c_pdf, c_share = st.columns(2)
            
            with c_pdf:
                with st.spinner("📄 Generating PDF..."):
                    try:
                        pdf_bytes = generate_pdf_report(plant_state)
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"floravision_{plant_state.plant_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            width="stretch"
                        )
                    except Exception as pdf_error:
                        st.warning(f"PDF failed: {pdf_error}")
            
            with c_share:
                # Share via Email (mailto)
                subject = f"FloraVision AI Diagnosis: {plant_state.plant_name.replace('_', ' ').title()}"
                body = f"Hello,\n\nI just diagnosed my {plant_state.plant_name.replace('_', ' ').title()} using FloraVision AI.\n\nStatus: {severity}\n\nSummary:\n{sections[0] if sections else 'Check attachment'}\n\nBuilt with FloraVision AI 🌿"
                import urllib.parse
                mailto_link = f"mailto:?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
                
                st.markdown(f"""
                    <a href="{mailto_link}" class="share-btn">
                        ✉️ Share Report via Email
                    </a>
                """, unsafe_allow_html=True)

            # 💬 FOLLOW-UP CHAT
            st.divider()
            st.subheader("💬 Ask a Botanist")
            st.info("Have more questions? Ask about specific care steps or pet safety.")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ex: Is this plant safe for my cat?"):
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Botanist is thinking..."):
                        response = chat_manager.get_response(
                            prompt, 
                            plant_state, 
                            st.session_state.messages[:-1]
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun() # Ensure the new message is displayed immediately

else:
    # 📊 HISTORY DASHBOARD PAGE
    st.markdown("<h1 class='main-header'>📊 History Dashboard</h1>", unsafe_allow_html=True)
    
    stats = db_manager.get_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scans", stats["total_diagnoses"])
    col2.metric("Healthy Plants", stats["healthy_plants"])
    col3.metric("Issues Detected", stats["total_diagnoses"] - stats["healthy_plants"])
    
    st.divider()
    
    history = db_manager.get_history()
    
    if not history:
        st.info("No diagnosis history yet. Start by scanning a plant!")
    else:
        # Show progression chart
        if len(history) > 1:
            with st.expander("📈 Health Trends Over Time", expanded=True):
                # Simple count of diagnoses by date
                dates = [h['timestamp'].split('T')[0] for h in history]
                date_counts = {}
                for d in dates:
                    date_counts[d] = date_counts.get(d, 0) + 1
                
                # Convert to format for st.line_chart
                chart_data = {
                    "Date": list(date_counts.keys())[::-1],
                    "Scan Volume": list(date_counts.values())[::-1]
                }
                st.line_chart(chart_data, x="Date", y="Scan Volume")

        st.subheader("Recent Diagnoses")
        
        for record in history:
            timestamp = datetime.fromisoformat(record['timestamp']).strftime('%Y-%m-%d %H:%M')
            severity = record['severity'] or "Healthy"
            plant = record['plant_name'].replace('_', ' ').title()
            
            with st.expander(f"{'🟢' if record['is_healthy'] else '🔴'} {timestamp} - {plant} ({severity})"):
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    if record['image_path'] and os.path.exists(record['image_path']):
                        st.image(record['image_path'], width="stretch")
                    else:
                        st.warning("Image not found")
                
                with col_info:
                    st.markdown(f"**Severity:** {severity}")
                    st.markdown(f"**Confidence:** {record['confidence']}")
                    if st.button(f"View Full Report #{record['id']}", key=f"view_{record['id']}"):
                        st.session_state.view_report_id = record['id']
                        st.rerun()
                
                if st.button(f"🗑️ Delete Record", key=f"del_{record['id']}", type="secondary"):
                    db_manager.delete_diagnosis(record['id'])
                    st.toast("Record deleted")
                    st.rerun()

    # Detail View overlay if requested
    if 'view_report_id' in st.session_state:
        report_data = db_manager.get_diagnosis_by_id(st.session_state.view_report_id)
        if report_data:
            st.divider()
            st.markdown(f"### Historical Report for {report_data['plant_name'].title()}")
            st.markdown(report_data['final_response'])
            if st.button("Close Report"):
                del st.session_state.view_report_id
                st.rerun()

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8rem;'>"
    "FloraVision AI • Made with 🌱 for healthier plants"
    "</p>",
    unsafe_allow_html=True
)
