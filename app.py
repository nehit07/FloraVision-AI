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

# Import FloraVision components
from src.floravision.graph import run_diagnosis, run_diagnosis_full
from src.floravision.nodes.seasonal import get_season_from_month
from src.floravision.utils.pdf_report import generate_pdf_report
from src.floravision.state import PlantState


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FloraVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
    }
    .stImage {
        border-radius: 1rem;
    }
    div[data-testid="stMarkdownContainer"] h2 {
        margin-top: 1.5rem;
    }
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
    
    # Mode toggle
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
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

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
    st.subheader("📁 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of the plant you want to diagnose"
    )

st.divider()

# Process the image
image_bytes = None
image_source = None

if camera_image is not None:
    image_bytes = camera_image.getvalue()
    image_source = "camera"
elif uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_source = "upload"

# Show image preview and analyze button
if image_bytes:
    col_preview, col_analyze = st.columns([1, 2])
    
    with col_preview:
        st.image(image_bytes, caption="Ready to analyze", width="stretch")
    
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
            use_container_width=True
        )

    if analyze_button:
        with st.spinner("🔍 Analyzing your plant... This may take a moment..."):
            try:
                # Run the diagnosis pipeline (get full state for PDF)
                result = run_diagnosis_full(
                    image_bytes=image_bytes,
                    season=season,
                    mock=mock_mode
                )
                
                # Create PlantState object for PDF generation
                plant_state = PlantState(**result)
                
                st.success("✅ Analysis complete!")
                st.divider()
                
                # Display the diagnosis
                st.markdown("# 🌿 Diagnosis Results")
                st.markdown(result["final_response"])
                
                # Generate PDF report
                st.divider()
                
                with st.spinner("📄 Generating PDF report..."):
                    try:
                        pdf_bytes = generate_pdf_report(plant_state)
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"floravision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as pdf_error:
                        # Fallback to markdown if PDF fails
                        st.warning(f"PDF generation failed: {pdf_error}. Offering markdown instead.")
                        st.download_button(
                            label="📥 Download Diagnosis (Markdown)",
                            data=result["final_response"],
                            file_name=f"floravision_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Try enabling Demo Mode in the sidebar, or check your API key configuration.")

else:
    # Show instructions when no image is provided
    st.info("👆 **Get started:** Capture a photo with your camera or upload an image of your plant above.")
    
    # Example section
    with st.expander("💡 Tips for best results"):
        st.markdown("""
        1. **Good lighting** - Natural daylight works best
        2. **Focus on problem areas** - If you see spots or discoloration, include them
        3. **Capture both leaves and stems** - More information helps diagnosis
        4. **Avoid blurry images** - Hold steady or use a flat surface
        5. **Multiple angles** - For complex issues, analyze multiple photos
        """)


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
