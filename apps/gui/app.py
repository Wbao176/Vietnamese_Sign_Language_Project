import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import mediapipe as mp
mp_holistic = mp.solutions.holistic
import json

import cv2
import torch
from notebooks.archive.GUI.utils import get_rotation_demo, get_scaling_demo, get_time_stretch_plot, get_translation_demo

from notebooks.archive.GUI.demo import mediapipe_detection, extract_keypoints, interpolate_sequence, extract_live_phonetic_features
from notebooks.archive.GUI.SA_CNN_LSTM_model import MultiViewSAConvlLSTM


# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VSL System Dashboard",
    page_icon="🎯",
    layout="wide",
) 

# --- 2. Data Preparation ---

# Dataset Split Table Data
split_data = {
    "Split": ["1", "2", "3", "4", "5", "6", "7", "**Total**"],
    "Video ID Range": ["000000 - 003540", "003541 - 007081", "007082 - 010617", "010618 - 014158", "014159 - 017699", "017700 - 021238", "021239 - 024774", "000000 - 024774"],
    "Videos per View": ["3,541", "3,541", "3,536", "3,541", "3,541", "3,539", "3,536", "**24,775**"],
    "Total Videos": ["10,623", "10,623", "10,608", "10,623", "10,623", "10,617", "10,608", "**74,325**"]
}
df_split = pd.DataFrame(split_data)

# Semantic Groups (from image_48eb3a.png)
group_data = pd.DataFrame({
    "Group": ["Group 1: Human and Family", "Group 2: Occupation and Places", "Group 3: Objects and Items", 
              "Group 4: Transport and Food", "Group 5: Time, Nature and Colors", "Group 6: Activities, Adjectives and Concepts"],
    "Percentage": [5.55, 9.14, 14.87, 7.41, 19.52, 43.51],
    "Count": ["4,125", "6,786", "11,046", "5,502", "14,493", "32,307"],
    "Color": ["#4c6ef5", "#51cf66", "#fa5252", "#7950f2", "#fab005", "#862e2e"]
})

# Top 10 Signs (from image_3f013a.png)
top_signs_df = pd.DataFrame({
    "Sign": ["Mùa hè", "Ngày", "Nắng", "Ướt", "Nhẹ", "Anh", "Em", "Thơm", "Chị", "Rửa tay"],
    "Count": [279, 267, 261, 258, 255, 252, 252, 246, 243, 243]
})

# Box Plot Variance (from image_3ef252.png)
box_glosses = ["Ngày Nhà giáo Việt Nam", "Ngày Quốc tế Phụ nữ", "Ngày Quốc tế Lao động", "Ngày", "Mùa khô", "Trường Đại học", "Quạt (đứng)", "Nhẹ", "Mùa hè", "Ngày Quốc tế Thiếu nhi", "Rửa tay", "Máy điều hòa", "Cửa sổ", "Giường", "Máy giặt", "Cái cửa", "Bình minh", "Chị", "Tàu hỏa", "Ướt"]
box_records = []
np.random.seed(42)
for gloss in box_glosses:
    base = np.random.uniform(2.2, 4.0)
    samples = np.random.normal(loc=base, scale=0.3, size=50)
    for s in samples:
        box_records.append({"Word (Gloss)": gloss, "Video Length (seconds)": s})
df_box = pd.DataFrame(box_records)

# Video Length Histogram
hist_durations = np.random.normal(loc=2.52, scale=0.45, size=74259)
hist_durations = np.clip(hist_durations, 1.2, 5.8)
df_hist = pd.DataFrame({"Duration": hist_durations})

# Signer ID Distribution
signer_df = pd.DataFrame({
    "Signer ID": ["026", "024", "015", "025", "009", "020", "011", "014", "010", "023", "022", "001", "004", "016", "021"],
    "Number of Videos": [5550, 5280, 3450, 3150, 3080, 2980, 2800, 2760, 2650, 2610, 2610, 2580, 2560, 2550, 2540]
})

# --- 3. Sidebar Navigation ---
with st.sidebar:
    st.title("🎯 VSL System")
    main_page = st.radio("Main Menu", ["VSL400 Dataset", "Model Architecture", "Live Demo"])
    
    # Sub-navigation for Dataset section
    if main_page == "VSL400 Dataset":
        st.divider()
        st.markdown("### 📊 Dataset Sub-Menu")
        sub_page = st.selectbox("Select View", ["Exploratory Data Analysis (EDA)", "Preprocessing"])
    
    st.divider()
    st.info("VSL Recognition Project 2026")

# --- 4. Main Page Logic ---

if main_page == "VSL400 Dataset":
    if sub_page == "Exploratory Data Analysis (EDA)":
        st.title("VSL400 Dataset Explorer - EDA")
        
        # Header Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Glosses", "400")
        m2.metric("Total Videos", "74,259")
        m3.metric("Unique Signers", "28")
        m4.metric("Avg. Duration", "2.52s")
        st.divider()

        # Dataset Split Table
        st.subheader("📊 Dataset Split Overview")
        st.table(df_split)
        st.divider()

        # Box Plot
        st.subheader("📏 Video Length Variance per Gloss")
        fig_box = px.box(df_box, x="Word (Gloss)", y="Video Length (seconds)", color_discrete_sequence=["#add8e6"])
        fig_box.update_layout(plot_bgcolor="white", height=450, xaxis=dict(tickangle=-45))
        st.plotly_chart(fig_box, width='stretch')
        st.divider()

        # Semantic Groups & Top Signs
        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("📂 Semantic Group Distribution")
            fig_g = px.bar(group_data, x="Percentage", y="Group", orientation='h', color="Group", color_discrete_map={row["Group"]: row["Color"] for _, row in group_data.iterrows()})
            fig_g.update_layout(showlegend=False, plot_bgcolor="white", height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_g, width='stretch')
        with c_right:
            st.subheader("🔝 Top 10 Signs by Count")
            fig_s = px.bar(top_signs_df, x="Count", y="Sign", orientation='h', color_discrete_sequence=["#4c78a8"])
            fig_s.update_layout(plot_bgcolor="white", height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_s, width='stretch')
        st.divider()

        # Histogram
        st.subheader("📈 Frequency of Video Lengths")
        fig_h = px.histogram(df_hist, x="Duration", nbins=60, color_discrete_sequence=["#add8e6"])
        fig_h.update_layout(plot_bgcolor="white", xaxis_title="Video Length (seconds)", yaxis_title="Hz")
        fig_h.update_traces(marker_line_color='black', marker_line_width=0.5)
        st.plotly_chart(fig_h, width='stretch')
        st.divider()

        # Signer ID
        st.subheader("👤 Number of Videos per Signer ID")
        fig_signer = px.bar(signer_df, x="Signer ID", y="Number of Videos", color_discrete_sequence=["#ffc107"])
        fig_signer.update_layout(plot_bgcolor="white", xaxis=dict(type='category', tickangle=-45), height=450)
        st.plotly_chart(fig_signer, width='stretch')
        st.divider()

        # Multi-Angle View Samples
        st.subheader("🎥 Multi-Angle View Samples")
        st.info("💡 Each camera angle (Left, Front, Right) contains exactly **24,753** videos.")
        i1, i2, i3 = st.columns(3)
        with i1: st.image("left.png", caption="Left View (24,753 videos)", width='stretch')
        with i2: st.image("front.png", caption="Front View (24,753 videos)", width='stretch')
        with i3: st.image("right.png", caption="Right View (24,753 videos)", width='stretch')

    elif sub_page == "Preprocessing":
        st.title("VSL400 Dataset - Preprocessing")
        st.write("This section details the steps taken to prepare the raw video data for the recognition model.")
        st.image("preprocessing.png", caption="Preprocessing Pipeline", width='stretch')
        
        step_col1, step_col2 = st.columns(2)
        with step_col1:
            st.info("### 1. Frame Extraction")
            st.write("- Sampling rate: 30 FPS")
            st.write("- Resolution: 1080")
            st.image("frame.png", caption="Visualizing Sequence Sampling", width='stretch')

        with step_col2:
            st.info("### 2. Pose Estimation")
            st.write("- MediaPipe integration")
            st.write("- Tracking 42 hand landmarks (21 landmarks each hand) & 25 body points")
            st.image("pose.png", caption="Visualizing Pose Estimation", width='stretch')

            
        st.divider()

        # Row 2: Feature Extraction (New Section)
        st.info("### 3. Feature Extraction")
        feat_col1, feat_col2 = st.columns([1, 1])
        
        with feat_col1:
            st.markdown("""
            **Body Pose Landmarks:**
            - **Extraction**: Using MediaPipe Holistic to extract 25 landmark coordinates (x, y, z) for body (excluding below hip landmarks).
            - **Normalization**: Normalizing the landmark coordinates to a standard scale [0, 1].
            """)
            st.image("pose_landmarks.png", caption="MediaPipe Body Pose Landmarks", width='stretch')


        
        with feat_col2:
            st.markdown("""
            **Hands Landmarks:**
            - **Extraction**: Using MediaPipe Hands to extract 21 landmark coordinates (x, y, z) for each hand.
            - **Normalization**: Normalizing the landmark coordinates to a standard scale [0, 1].
            """)
            st.image("hand_landmarks.png", caption="MediaPipe Hands Landmarks", width='stretch')


        st.divider()
        st.markdown("""
            **Sequences:**
            - **Concatenation**: Combining 25 body pose landmarks and 42 hands pose landmarks into an 1D array of coordinates with the shape [(x1, y1, z1, ..., z67)] (201 : 67 features x 3 dimensions).
            - **Feature Vector**: Mapping every 201 coordinates with the number of frames to get the final vector (frames, 201).
            - **Interpolation**: Fill missing values in the feature vectors to make all sequences have the same 60 frames.
            - **Final Shape**: The final input shape for the model is (60, 201) representing 60 frames and 201 features per frame.
            - **Saving Format**: Storing the preprocessed feature vectors in NumPy format (.npz) for efficient loading during model training and inference. 
            """)
        st.divider()
        st.info("### 3. Data Splitting")
        st.image("split.png", width='stretch')
        # Using custom CSS to inject a vertical line between the columns
        st.markdown(
            """
            <style>
            [data-testid="column"]:nth-child(2) {
                border-left: 2px solid #f0f2f6;
                padding-left: 25px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        split_col1, split_col2 = st.columns([1, 1], gap="large")
        
        with split_col1:
            st.markdown("""
            **Signer-Independent Split:**
            To ensure the model generalizes well to new users, we perform a distribution-aware split based on **Signer IDs** rather than random video selection, ensuring no signer appears in more than one set (train, val, test).
            
            - **Training (70%)**: Model learns core sign movements from the majority of signers.
            - **Validation (15%)**: Used for hyperparameter tuning on signers the model hasn't seen during training.
            - **Testing (15%)**: Final evaluation on completely unseen individuals to simulate real-world performance.
            """)

        with split_col2:
            # Simplified visualization of the logic in your code
            st.markdown("""
            **Splitting Logic:**
            1. Group dataset by **Gloss** (Vocabulary word).
            2. For each word, shuffle the list of unique **Signer IDs**.
            3. Distribute signers across sets to maintain a balanced vocabulary in Train, Val, and Test.
            4. Export final metadata as `train.csv`, `val.csv`, and `test.csv`.
            """)


        st.divider()
        st.info("### 4. Data Augmentation")
        st.image("augmentation.png" , width='stretch')
        st.markdown("""
        To ensure the model is robust against different camera angles, distances, and signing speeds, we apply 
        five key geometric and temporal transformations to the keypoint sequences.
        """)

        # Create three columns for the core augmentation categories
        scale_col1, scale_col2 = st.columns(2, gap="large")

        with scale_col1:
            st.markdown("#### 📏 Spatial Scaling")
            st.write("""
            - **Function**: `scale_sequence_function`
            - **Logic**: Randomly zooms the skeleton (70% - 126%).
            - **Goal**: Distance invariance (Signer close vs. far).
            """)

            st.write(
                    "The algorithm of this function is **Centroid-Based Homothety**, "
                    "which implements a spatial scaling transformation to simulate varying "
                    "subject-to-camera distances."
                )
                
    # Display the Centroid calculation
            st.write("The algorithm first identifies a dynamic centroid:")
            st.latex(r"\text{Centroid} = (\bar{x}, \bar{y})")
            st.write("Where:")
            st.latex(r"\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i \quad \text{and} \quad \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i")
            st.write("are the average $x$ and $y$ coordinates of all landmarks in the frame, respectively.") 
            st.write("The scaling transformation is then applied to each landmark $P_i$ with s in range (0.7, 1.26) as follows:")                       
            st.latex(r"P'_i = (P_i - \text{Centroid}) \times s + \text{Centroid}")
            st.write("Where $P_i$ is the original landmark, $s$ is the scaling factor, and Centroid is the average position of all landmarks in the frame.")
            
        with scale_col2:
                               # Scaling Slider (based on your code's 0.7 to 1.26 range)
            scale_val = st.slider("Preview Scale:", 0.5, 1.5, 1.0, key="scale_slider")
            
            # Display Scaled Hand
            fig_scale = get_scaling_demo(scale_val)
            st.plotly_chart(fig_scale, use_container_width=True)

        
        st.divider()

        rotate_col1, rotate_col2 = st.columns(2, gap="large")

        with rotate_col1:

            st.markdown("#### 🔄 Geometric Rotation")
            st.write("""
                - **Function**: `rotate_sequence_function`
                - **Logic**: Tilts landmarks by ±15°.
                - **Goal**: Robustness against tilted camera angles.
                """)
            
            st.write(
            "By applying the **2D Rotation Matrix Transformation** algorithm, this function "
            "simulates camera tilt or slight signer leaning by rotating the skeletal structure "
            "within a range of $\pm15^\circ$."
            )
            st.latex(r"""
            \begin{bmatrix} x' \\ y' \end{bmatrix} = 
            \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}
            \begin{bmatrix} x - x_c \\ y - y_c \end{bmatrix} + 
            \begin{bmatrix} x_c \\ y_c \end{bmatrix}
            """)
            
            st.write(
                "The function calculates a rotation matrix based on a randomly sampled angle $\\theta$, "
                "using the median of valid pose landmarks $(x_c, y_c)$ as the pivot point."
            )
            
            # The Goal/Benefit
            st.info(
                "**Result**: This prevents the model from over-fitting perfectly vertical postures, "
                "a common bias in laboratory-collected sign language datasets."
            )


        with rotate_col2:

            rot_angle = st.slider("Preview Rotation:", -90, 90, 30, key="hand_rot")
            fig_hand = get_rotation_demo(rot_angle)
            st.plotly_chart(fig_hand, width='stretch')

        st.divider()
        st.info("### ➕ Translation Function")
        t_col1, t_col2 = st.columns([1, 1])

        with t_col1:
            st.write(
                "This function applies a **global offset** to the skeletal coordinates "
                "to simulate off-center positioning based on the **Stochastic Linear Shifting** algorithm."
            )
            
            # Mathematical Representation
            st.latex(r"P'_i = P_i + \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}")
            
            st.write(
                "The mechanism includes sampling a random displacement $(\Delta x, \Delta y)$ "
                "from a Gaussian-like range $[-0.5, 0.5]$ and adding it to any non-zero landmark."
            )
            
            # Clipping Rule
            st.warning(
                "**Data Validity**: A clipping operation is executed to ensure all translated "
                "points remain in the range $[0.0, 1.0]$."
            )

        with t_col2:
            # Interactive Sliders for Translation
            off_x = st.slider("Horizontal Shift (Δx):", -0.5, 0.5, 0.1)
            off_y = st.slider("Vertical Shift (Δy):", -0.5, 0.5, -0.1)
            
            fig_trans = get_translation_demo(off_x, off_y)
            st.plotly_chart(fig_trans, use_container_width=True)

        st.divider()

        temp_col1, temp_col2 = st.columns(2, gap="large")

        with temp_col1:
            st.markdown("#### ⏱️ Temporal Stretching")
            st.write("""
            - **Function**: `time_stretch_function`
            - **Logic**: Resamples sequences (±20% speed).
            - **Goal**: Robustness against different signing paces.
            """)
            st.write(
            "Utilizing the **Nearest-Neighbor Temporal Resampling** algorithm, which modifies "
            "the frame rate of the sequence by a factor $v \in [0.8, 1.2]$."
            )

            # Formula matching your specific Python logic: new_frames = org_frames / v
            st.latex(r"N_{new} = \text{round}\left(\frac{N_{old}}{v}\right)")
            
            # Formula for the index mapping used in your np.linspace
            st.latex(r"Index_{map} = \text{round}(\text{linspace}(0, N_{old}-1, N_{new}))")
            
            st.write(
                "This algorithm resamples frames based on the calculated index map to simulate "
                "varying signing speeds while preserving motion characteristics."
            )
        with temp_col2:
           # Rate typically ranges from 0.8x to 1.2x
            rate = st.slider("Speed Factor (Rate):", 0.8, 1.2, 1.0, step=0.05)
            
            st.plotly_chart(get_time_stretch_plot(rate), use_container_width=True)
            
            st.info("💡 **Fast Signing**: The curve finishes earlier than **Slow Signing** ")
        
        st.divider()

        st.markdown("#### 👐 Inter-Hand Distance")
            
        st.write(
                "This function simulates variations in signing width by adjusting the distance "
                "between the left and right wrists by first determines the new target positions for the wrists before solving for the elbows."
            )

        st.write(""" **Current Mid-Point (x):** """)
        st.latex(r"x_{mid} = \frac{x_{left_wrist} + x_{right_wrist}}{2}")
        st.write(""" **Current Distance (dx):** """)
        st.latex(r"d_{current} = |x_{right_wrist} - x_{left_wrist}|")
        st.write(""" **Target Distance (dx):** """)
        st.latex(r"d_{target} = d_{current} + \Delta dx")
            
        st.write(
                "where $\Delta dx$ is the random change sampled from (-0.1, 0.1)."
            )
        st.write("**Finally, the new target $x$ coordinates for both wrists are calculated:**")
            
            # Target positions formula from the image
        st.latex(r"x'_{left} = x_{mid} - \frac{d_{target}}{2} \quad \text{and} \quad x'_{right} = x_{mid} + \frac{d_{target}}{2}")
            
        st.write(
                "These target coordinates $x'_{left}$ and $x'_{right}$ are then passed to the "
                "**Inverse Kinematics** solver to find the corresponding elbow positions."
            )

        st.divider()
        st.markdown("#### 👐 Inverse Kinematic Function")
        ik_left , ik_right = st.columns(2)
        with ik_left:
            st.write(
                "The IK solver uses the **Analytical Two-Link Arm Model** to calculate the elbow position based on the shoulder and wrist coordinates."
            )
            st.write("Once the new wrist target $P_{wrist}'$ is set, the inverse_kinematics_function calculates the elbow position ($P_{elbow}$) to maintain anatomical arm lengths ($L_1$ for arm, $L_2$ for forearm).")
            st.write(""" **Distance to Target(d):** """)
            st.latex(r"d = \text{dist}(P_{shoulder}, P'_{wrist})")
            st.write(
                    "Where $d$ is the distance from shoulder to wrist target. The solver selects "
                    "between two possible elbow solutions based on the **original bend direction** "
                    "to maintain natural human posture."
                )            
            st.write(""" **Distance from Shoulder to Projection Point (a):** """)
            st.latex(r"a = \frac{d^2 + L_1^2 - L_2^2}{2d}")
            st.write(""" **Elbow Height (h):** """)
            st.latex(r"h = \sqrt{L_1^2 - a^2}")
            st.write("""**Elbow Position Calculation:** """)
            st.latex(r"P_{elbow} = P_{shoulder} + a \cdot v_{wrist} \pm h \cdot v_{perp}")
        with ik_right:
            st.image("arm.png", caption="Visualizing Inverse Kinematics Adjustment", width='stretch')
            
        st.divider()

        # --- Data Integrity Guardrails ---
        st.markdown("#### 🛡️ Transformation Guardrails")
        guard_1, guard_2, guard_3 = st.columns(3)
        guard_1.warning("**NaN/Inf Protection**: Invalid coordinates are automatically reverted to original frames.")
        guard_2.warning("**Coordinate Clipping**: Ensures all $(x, y)$ remain within the [0, 1] screen space.")
        guard_3.warning("**Zero-Masking**: Augmentation ignores landmarks not detected by MediaPipe.")



elif main_page == "Model Architecture":
    st.title("🏗️ Model Architecture")
    st.write("Explore the deep learning frameworks used to translate VSL400 skeletal data into text.")

    # Sub-page navigation
    sub_tab = st.radio(
        "Select Architecture View:",
        ["Overall Structure", "SA-CNN-LSTM", "ST-GCN"],
        horizontal=True
    )

    st.divider()

    # --- 1. Overall Structure ---
    if sub_tab == "Overall Structure":
        st.subheader("🌐 High-Level Pipeline")
        st.write(
            "The system follows a modular flow: from raw video input to skeletal landmark "
            "extraction, followed by spatial-temporal feature learning."
        )
        st.image("architecture.png", caption="Overall Model Architecture", width='stretch')
        
        # Conceptual columns for the flow
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**1. Input Module**\n\nMediaPipe extracts 21 points per hand and 33 body points.")
        with c2:
            st.info("**2. Processing Module**\n\nAugmentation (IK, Scaling) and Normalization.")
        with c3:
            st.info("**3. Recognition Module**\n\nParallel processing via CNN-LSTM and Graph Networks.")

    # --- 2. SA-CNN-LSTM ---
    elif sub_tab == "SA-CNN-LSTM":
        st.subheader("🧠 SA-CNN-LSTM")
        st.write(
        "The SA-CNN-LSTM (Self-Attention Convolutional LSTM) is a hybrid architecture designed "
        "to extract linguistic meaning from skeletal motion through a three-stage pipeline."
    )

    # --- STAGE 1: PHONETIC FEATURE EXTRACTION ---
        st.divider()
        st.subheader("1️⃣ Phonetic Feature Extraction")
        st.write(
            "Before the neural network, raw $(x, y, z)$ landmarks are transformed into "
            "high-level phonetic descriptors to ensure the model focuses on linguistic mechanics."
        )
        left, center, right = st.columns([1, 1, 1])
        with center:
            st.image("phonetic.png", caption="Hand Shape and Orientation Features", width=400)

        feat_col1, feat_col2 = st.columns(2, gap="large")
        with feat_col1:
            st.markdown("**Hand Shape & Orientation**")
            st.write("* **Finger Flexion:** Calculate the bend of each finger by using dot product of bone vectors.")
            st.latex(r"\beta_(ij) = \arccos\left(\frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|}\right)")

            st.write("* **Inter-Finger Spread:** 4 angles measuring finger separation.")
            st.latex(r"\omega_i = \arccos\left(\frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|}\right)")

            st.write("* **Palm Orientation:** A 3D vector ($n_x, n_y, n_z$) derived from the cross product of the index and pinky metacarpal vectors, defining which way the hand faces.")
            st.latex(r"\vec{v}_1 = P_{index\_mcp} - P_{wrist}")
            st.latex(r"\vec{v}_2 = P_{pinky\_mcp} - P_{wrist}")
            st.latex(r"\vec{n}_{raw} = \vec{v}_1 \times \vec{v}_2")
            st.latex(r"\vec{n}_{palm} = \frac{\vec{n}_{raw}}{\|\vec{n}_{raw}\| + 1e^{-6}}")
            st.write("* **Hand Location:** The (x, y, z) coordinates of the middle finger MCP joint relative to the wrist.")

        
        with feat_col2:


            st.markdown("**Motion Dynamics**")
            st.write("* **Velocity:** This represents the first derivative of the hand's position, capturing the change in coordinates between the previous and current frames")
            st.latex(r"v_t = \frac{p_{curr} - p_{prev}}{\Delta t}")
            st.write("* **Acceleration:** This represents the second derivative, capturing the rate of change of velocity and identifying sudden movements or stops.")
            st.latex(r"a = \frac{p_{next} - 2p_{curr} + p_{prev}}{\Delta t}")
            st.write("* **Curvature ($\kappa$):** Identifies arcs and circular paths.")
            st.latex(r"\kappa = \frac{4 \times \text{Area}(p_{t-1}, p_t, p_{t+1})}{d_1 d_2 d_3}")

        st.info("Features are all concatenated into an vector (60, 64) representing 60 frames and 64 phonetic features per frame, which is then fed into the ConvLSTM for spatial-temporal processing.")

        # --- STAGE 2: SPATIAL-TEMPORAL PROCESSING ---
        st.divider()
        st.subheader("2️⃣ Convolutional LSTM & Self-Attention")
        st.write(

            "The phonetic grid is processed through a **ConvLSTM** cell. Unlike standard LSTMs, "
            "it uses convolutions ($*$) to preserve the spatial relationship of the $8*8$ phonetic features."
        )

        st.image("sa_cnn_lstm.png", caption="ConvLSTM Cell with Self-Attention", width='stretch')

        

        st.write("* **Input gate ($i_t$):**" )

        st.latex(r"i_t = \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + b_i)")
        st.write("where $W_{xi}$ and $W_{hi}$ are convolutional kernels applied to the current input and previous hidden state, respectively, and $b_i$ is the bias term. The sigmoid function ($\sigma$) ensures that the gate outputs values between 0 and 1, determining how much of the new information should be added to the cell state.")

        st.write("**Purpose:** This gate decides which new information from the current frame is worth keeping." )

        st.write("**Mechanics:** It performs a convolution on the current phonetic input ($\mathcal{X}_t$) and the previous hidden state ($\mathcal{H}_{t-1}$). The sigmoid function ($\sigma$) squashes the values between 0 (ignore everything) and 1 (keep everything)." )


        st.write("* **Forget gate ($f_t$):**" )

        st.latex(r"f_t = \sigma(W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + b_f)")
        
        st.write("**Purpose:** This gate determines what information from the past is no longer relevant and should be forgotten." )
        st.write("**Mechanics:** Similar to the input gate, it applies convolutional operations to the current input and previous hidden state, followed by a sigmoid activation to produce a forget vector that scales the previous cell state ($\mathcal{C}_{t-1}$). Values close to 0 will effectively erase that information, while values close to 1 will retain it." )
        
        st.write("* **The Cell State ($\mathcal{C}_t$) - Long-Term Memory:**" )
        st.latex(r"\mathcal{C}_t = f_t \odot \mathcal{C}_{t-1} + i_t \odot \tanh(W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c)")
        st.write("**Purpose:** The cell state serves as the long-term memory of the LSTM, carrying information across time steps. It is updated by combining the retained past information (scaled by the forget gate) and the new candidate values (scaled by the input gate).")
        st.write("**Mechanics:** The previous cell state ($\mathcal{C}_{t-1}$) is multiplied element-wise by the forget vector ($f_t$) to determine what to retain. Simultaneously, a new candidate cell state is computed using a convolutional operation on the current input and previous hidden state, passed through a tanh activation to ensure values are between -1 and 1. This candidate is then scaled by the input gate ($i_t$) to determine how much new information to add. The final cell state is the sum of these two components." )

        st.write("* **Output gate ($o_t$) and Hidden State ($\mathcal{H}_t$):**" )
        st.latex(r"\mathcal{H}_t = o_t \odot \tanh(\mathcal{C}_t)")

        st.write("**Purpose:** The output gate controls how much of the cell state should be exposed to the next layer (Self-Attention layer) or time step. The hidden state ($\mathcal{H}_t$) is the output of the LSTM cell at time $t$, which is influenced by both the current cell state and the output gate .")
        st.write("**Mechanics:** The updated cell state ($\mathcal{C}_t$) is passed through a $\ tanh$ function and then filtered by the Output Gate ($o_t$,  it follows the same logic as $i_t$ and $f_t$)." )
        

        st.markdown("* **Spatial Self-Attention Mechanism**")
        st.write(
            "At each time step, the function identifies which phonetic features "
            "are most critical for the current sign (e.g., prioritizing finger bend over elbow velocity)."
            "Sign language often relies on a Dominant Feature (e.g., only the thumb moving). The attention mechanism allows the model to ignore 90% of the hand and focus only on the moving part."
        )
        
        # Image Reference: image_506598.png logic for attention projection
        att_col1, att_col2 = st.columns(2, gap="large")
        with att_col1:
            st.latex(r"S = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)")
            st.write("Where $Q$, $K$, and $V$ are linear projections of the hidden state ($\mathcal{H}_t$) and $d_k$ is the dimensionality of the key vectors. The softmax function normalizes the attention scores to create a probability distribution over the features.")
        with att_col2:
            st.latex(r"H_{out} = \gamma(S \cdot V) + H_{in}")
            st.write(" Where $\gamma$ is a learnable parameter that balances the attention map with the original features.")

        st.write("Residual Output ($\hat{H}_t$): The attended features are added back to the original state to ensure no data is lost during the focusing process")
        # --- STAGE 3: MULTI-VIEW FUSION ---
        st.divider()
        st.subheader("3️⃣ Multi-View Late Fusion")
        st.write(
            "To resolve occlusions, features are extracted from three camera perspectives "
            "(Front, Left, Right) and fused before final classification."
        )
        st.image("multi.png", caption="Multi-View Late Fusion Strategy", width='stretch')

        st.success(
            "**Final Pipeline Output:** A synchronized 400-class prediction based on "
            "60 frames of multi-view phonetic data."
        )

        st.divider()
        st.subheader("📊 Performance Metrics")

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "77%")
        col2.metric("Precision", "79%")
        col3.metric("Recall", "77%")
        col4.metric("F1-Score", "76%")

        with st.expander("View Detailed Classification Report"):
            st.text("""
        ----------------- precision  recall  f1-score   support
        accuracy    -------------------------  0.77      2885
        macro avg          0.79      0.78      0.77      2885
        weighted avg       0.79      0.77      0.76      2885
            """)

        st.image("result_1.png", caption="Result on epochs", width='stretch')

    # --- 3. ST-GCN ---
    elif sub_tab == "ST-GCN":
        st.header("🧬 Spatio-Temporal Graph Convolutional Network")

        st.image("gcn.png", caption="ST-GCN Architecture Overview", width='stretch')
        
        # --- GRAPH STRUCTURE ---
        st.subheader("1️⃣ Physical Graph Representation")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("The hand landmarks are mapped to an Adjacency Matrix $A$.")
            st.latex(r"A \in \{0, 1\}^{3 \times 42 \times 42}")
        with col_b:
            st.info("Channels: [0] Self-loops, [1] Centripetal, [2] Centrifugal")
            st.write("This structure allows the model to learn both local and global spatial relationships between joints, as well as temporal dynamics across frames.")
            st.write("* **Channel 0 (Self-loops)**: Connects a joint to itself to maintain its own feature identity.")
            st.write("* **Channel 1 (Centripetal)**: Connects neighbor joints toward the root (the wrist).")
            st.write("* **Channel 2 (Centrifugal)**: Connects joints moving away from the root toward the fingertips.")



        # --- BLOCK LOGIC ---
        st.divider()
        st.subheader("2️⃣ Adaptive ST-GCN Block")
        st.image("block.png", caption="ST-GCN Block with Adaptive Convolution and SE Attention", width='stretch')
        
        st.markdown("**Adaptive Spatial Convolution:**")
        st.latex(r"x_{out} = \sum_{a \in \{0,1,2\}} (x_{in} \cdot W_a) \circledast (A_a \cdot M_a + B_a)")
        st.caption("Where $M$ is Edge Importance and $B$ is a Learnable Adaptive Graph.")

        st.markdown("**Squeeze-and-Excitation (Attention):**")
        st.latex(r"y = \text{Sigmoid}(FC(ReLU(FC(\text{AvgPool}(x)))))")
        st.latex(r"x_{se} = x \otimes y")

        # --- MULTI-STREAM FUSION ---
        st.divider()
        st.subheader("3️⃣ Multi-Stream Late Fusion")
        st.write("The model fuses three data modalities for maximum accuracy:")
        
        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("Joint Stream", "3 x 42")
        f_col2.metric("Bone Stream", "3 x 42")
        f_col3.metric("Motion Stream", "3 x 42")

        st.latex(r"\text{Score} = \text{Softmax}(\text{MLP}([\text{feat}_{joint} \oplus \text{feat}_{bone} \oplus \text{feat}_{motion}]))")
        st.success("Total Fusion Vector: 768 Dimensions")

        st.divider()
        st.subheader("📊 Performance Metrics")

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "75%")
        col2.metric("Precision", "77%")
        col3.metric("Recall", "75%")
        col4.metric("Macro F1-Score: 0.7538", "75%")

        with st.expander("View Detailed Classification Report"):
            st.text("""
        ----------------- precision  recall  f1-score   support
        accuracy    -------------------------  0.75      2892
         macro avg          0.77      0.76      0.75      2892
        weighted avg       0.77      0.75      0.75      2892
            """)

        st.image("result_2.png", caption="Result on epochs", width='stretch')


elif main_page == "Live Demo":

    st.set_page_config(page_title="SLR Live Demo", layout="wide")
    # --- LOAD MODEL ---
    @st.cache_resource
    def load_resources():
        model = MultiViewSAConvlLSTM(num_classes=400)
        model.load_state_dict(torch.load("best_vsl_model.pth", map_location=torch.device('cpu')))
        model.eval()
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return model, holistic

    model, holistic_model = load_resources()

    # --- UI LAYOUT ---
    st.title("🖐️ Real-Time Sign Language Recognition")
    run = st.toggle('Enable Camera')
    FRAME_WINDOW = st.image([])
    status_text = st.empty()

    # Buffers
    coord_buffer = []

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detection
        image, results = mediapipe_detection(frame, holistic_model)
        keypoints = extract_keypoints(results) # [3, 67]
        coord_buffer.append(keypoints)
        
        # Keep buffer at a reasonable size for interpolation (e.g., 30-40 frames)
        if len(coord_buffer) > 40:
            coord_buffer.pop(0)

        # 2. Inference (Triggered every 10 frames if buffer is sufficient)
        if len(coord_buffer) >= 30:
            # Interpolate to the 60 frames your model expects
            input_data = interpolate_sequence(coord_buffer, target_len=60)
            
            phonetic_input = extract_live_phonetic_features(input_data)
            with torch.no_grad():
                    # LateFusion_STGCN forward pass
                    output = model(phonetic_input.to(torch.device('cpu'))) 
                    
                    # Get prediction and confidence
                    probabilities = torch.softmax(output, dim=1)
                    confidence, prediction_idx = torch.max(probabilities, dim=1)
                    
                    if confidence.item() > 0.5: # Only show if model is confident
                        label_map = json.load(open("label_map.json"))
                        predicted_label = label_map[prediction_idx.item()]
                        status_text.success(f"Predicted Sign: **{predicted_label}** ({confidence.item():.2%})")
                    else:
                        status_text.warning("Analyzing motion...")
        else:
            status_text.info("Scanning for clear sign...")

        # 3. Update Visuals
        # Draw landmarks on the image here if desired using mp_drawing
        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not run:
        cap.release()
        status_text.info("Camera Stopped.")