import streamlit as st
import os
import google.generativeai as genai
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from cnn import Cnn
from io import BytesIO



def setup_gemini():
    genai.configure(api_key='AIzaSyDyG1I9Z0ZaID1EsSd9yBeaqxnWGFOC7sI')
    try:
        models = genai.list_models()  # Kiểm tra mô hình có sẵn
        print("Available models:", models)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        return model
    except Exception as e:
        print(f"Error listing models: {e}")
        return None


# Disease prediction using Gemini
def get_gemini_response(model, symptoms, lang="en"):
    if lang == "en":
        prompt = f"""
        Based on the following symptoms, suggest possible medical conditions and provide brief explanations:
        Symptoms: {symptoms}

        """
    else:  # Vietnamese
        prompt = f"""
        Dựa trên các triệu chứng sau đây, hãy đề xuất các tình trạng y tế có thể xảy ra và cung cấp giải thích ngắn gọn:
        Triệu chứng: {symptoms}

        """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating response: {str(e)}"


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize về kích thước phù hợp
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_path).unsqueeze(0)  # Thêm chiều cho ảnh


def load_model():
    # Sử dụng thiết bị phù hợp (CPU hoặc GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình Cnn với 4 lớp phân loại
    model = Cnn(num_classes=4).to(device)

    # Đường dẫn đến file checkpoint
    checkpoint_path = os.path.join("trained_model", "best_cnn.pt")

    if not os.path.exists(checkpoint_path):
        st.error(f"Không tìm thấy file model tại {checkpoint_path}")
        return None

    try:
        # Cách 1: Load với weights_only=False (cách an toàn nhất nếu bạn tin tưởng nguồn checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        try:
            # Cách 2: Thử thêm numpy.core.multiarray.scalar vào danh sách safe globals
            import numpy as np
            from torch.serialization import add_safe_globals

            # Thêm numpy.core.multiarray.scalar vào danh sách safe globals
            add_safe_globals([np.core.multiarray.scalar])

            # Thử load lại với weights_only=True (mặc định)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
        except Exception as inner_e:
            st.error(f"Không thể load model: {str(inner_e)}")
            return None

    # Chuyển sang chế độ dự đoán
    model.eval()

    return model


# Streamlit page config
st.set_page_config(page_title="Chatbot AI", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 5px 0; }
        .user { background-color: #dcf8c6; text-align: right; }
        .bot { background-color: #f1f1f1; }

        /* CSS để tự động cuộn xuống phần chat */
        #chat-container {
            max-height: 600px;
            overflow-y: auto;
        }

        /* Xóa padding dưới cho container */
        .block-container {
            padding-bottom: 2rem;
        }

        /* Make columns equal height */
        .main-columns {
            display: flex;
            min-height: 700px;
        }

        /* Add some space between columns */
        .column-gap {
            padding: 0 10px;
        }

        /* Language toggle styling */
        .language-toggle {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
        }

        .language-button {
            margin: 0 5px;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }

        .active-lang {
            background-color: #4CAF50;
            color: white;
        }

        .inactive-lang {
            background-color: #f1f1f1;
            color: black;
        }
    </style>

    <script>
        // JavaScript để tự động cuộn xuống cuối phần chat
        function scrollToBottom() {
            var chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        // Chạy hàm khi trang đã tải xong
        window.addEventListener('load', scrollToBottom);
    </script>
    """,
    unsafe_allow_html=True,
)

# Initialize language selection if not already set
if "language" not in st.session_state:
    st.session_state.language = "en"

# Check URL parameters for language change - PUT THIS BEFORE THE BUTTONS
query_params = st.query_params
if "lang" in query_params:
    if query_params["lang"] == "vn" and st.session_state.language != "vn":
        st.session_state.language = "vn"
        st.rerun()
    elif query_params["lang"] == "en" and st.session_state.language != "en":
        st.session_state.language = "en"
        st.rerun()


# Function to get text based on current language
def get_text(en_text, vn_text):
    return en_text if st.session_state.language == "en" else vn_text


# Call GPT
gemini_model = setup_gemini()

# Language toggle with buttons instead of HTML
col_spacer, lang_col = st.columns([10, 2])  # Tạo không gian rộng bên trái, chỉ sử dụng 2/12 không gian cho nút ngôn ngữ

with lang_col:
    # Tạo 2 cột bên trong cột ngôn ngữ để đặt các nút gần nhau
    en_col, vn_col = st.columns(2)
    with en_col:
        if st.button("🇬🇧 EN", type="primary" if st.session_state.language == "en" else "secondary"):
            st.session_state.language = "en"
            st.query_params["lang"] = "en"
            st.rerun()
    with vn_col:
        if st.button("🇻🇳 VN", type="primary" if st.session_state.language == "vn" else "secondary"):
            st.session_state.language = "vn"
            st.query_params["lang"] = "vn"
            st.rerun()

# Tiêu đề chính cho trang
st.markdown(
    f"<h1 style='text-align: center;'>💬 {get_text('Chatbot AI - Healthcare services', 'Chatbot AI - Dịch vụ chăm sóc sức khỏe')} 🇻🇳</h1>",
    unsafe_allow_html=True
)

# Create two main columns for the page layout
left_col, right_col = st.columns(2)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to get treatment recommendations based on current language
def get_treatment_recommendations(model, condition, lang="en"):
    if lang == "en":
        prompt = f"""
        Provide evidence-based treatment recommendations and care guidelines for a patient with {condition} diagnosis from an X-ray.

        Please format your response as follows:
        1. Brief explanation of the condition
        2. Standard treatment approaches
        3. Home care recommendations
        4. When to see a doctor
        5. Treatment details and the name of the medication that should be used


        Note: This is for informational purposes. Always consult healthcare professionals for proper medical advice.
        """
    else:  # Vietnamese
        prompt = f"""
        Cung cấp các khuyến nghị điều trị dựa trên bằng chứng và hướng dẫn chăm sóc cho bệnh nhân có chẩn đoán {condition} từ hình ảnh X-quang.

        Vui lòng định dạng phản hồi của bạn như sau:
        1. Giải thích ngắn gọn về tình trạng
        2. Phương pháp điều trị tiêu chuẩn
        3. Khuyến nghị chăm sóc tại nhà
        4. Khi nào nên gặp bác sĩ
        5. Chi tiết về cách điều trị và các tên của loại thuốc nên sử dụng

        Lưu ý: Đây là thông tin tham khảo. Luôn tham khảo ý kiến của chuyên gia y tế để được tư vấn y tế thích hợp.
        """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = "Error in generating treatment recommendations" if lang == "en" else "Lỗi khi tạo khuyến nghị điều trị"
        return f"{error_msg}: {str(e)}"


# LEFT COLUMN - Chat Interface
with left_col:
    st.subheader(get_text("Chat with AI 🖥️", "Trò chuyện với AI 🖥️"))

    # Display chat history with ID for JavaScript
    st.markdown('<div id="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Form để tránh auto-rerun khi nhập liệu
    with st.form(key="chat_form", clear_on_submit=True):
        placeholder_text = get_text(
            "Clearly state your symptoms here, please 🎀",
            "Vui lòng mô tả rõ ràng các triệu chứng của bạn ở đây 🎀"
        )
        input_text = st.text_input("", placeholder=placeholder_text, key="user_input")
        submit_button = st.form_submit_button(get_text("Send", "Gửi"))

    # Xử lý khi có tin nhắn mới
    if submit_button and input_text.strip():
        # Thêm tin nhắn người dùng vào session state
        user_prefix = "You:" if st.session_state.language == "en" else "Bạn:"
        st.session_state.messages.append({"role": "user", "content": f"**{user_prefix}** {input_text}"})

        # Generate AI response
        thinking_text = "🤖 Thinking..." if st.session_state.language == "en" else "🤖 Đang suy nghĩ..."
        with st.spinner(thinking_text):
            response = get_gemini_response(gemini_model, input_text, st.session_state.language)

        # Thêm phản hồi của AI vào session state
        ai_prefix = "AI:" if st.session_state.language == "en" else "AI:"
        st.session_state.messages.append({"role": "assistant", "content": f"**{ai_prefix}** {response}"})

        # Làm mới trang một lần để hiển thị tin nhắn mới
        st.rerun()

#=================================

# Add Grad-CAM Functions
def grad_cam(model, image, target_layer):
    # Ensure model is in eval mode
    model.eval()

    gradients = []
    activations = []

    # Register hooks to get gradients and activations
    def save_activation_hook(module, input, output):
        activations.append(output)

    def save_gradient_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks for target layer
    target_layer.register_forward_hook(save_activation_hook)
    target_layer.register_backward_hook(save_gradient_hook)

    # Forward pass through the model
    output = model(image)

    # Get the class with highest prediction probability
    class_idx = torch.argmax(output)

    # Backpropagate and calculate gradients
    model.zero_grad()
    output[0, class_idx].backward()

    # Get gradients and activations
    gradients = gradients[0].cpu().data.numpy()
    activations = activations[0].cpu().data.numpy()

    # Calculate Grad-CAM
    weights = np.mean(gradients, axis=(2, 3))  # Average gradients across width and height
    cam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):  # weights[0] because we only have 1 image in batch
        cam += w * activations[0, i, :, :]  # Summing weighted activations

    # Convert Grad-CAM to image and normalize
    cam = np.maximum(cam, 0)  # Remove negative values
    # Resize Grad-CAM to original input image size
    cam = cv2.resize(cam, (image.shape[3], image.shape[2]))  # Resize to (height, width) of input image
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    return cam, class_idx.item()


def generate_gradcam_image(model, image_tensor, target_layer, original_image):
    # Calculate Grad-CAM
    cam, class_idx = grad_cam(model, image_tensor, target_layer)

    # Convert PIL image to numpy array
    img_array = np.array(original_image)

    # Resize heatmap to match original image size
    cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))

    # Convert heatmap to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.5 + img_array
    superimposed_img = np.uint8(superimposed_img / np.max(superimposed_img) * 255)

    return superimposed_img, cam_resized, class_idx


# Modify the RIGHT COLUMN section in your deploy.py file to include Grad-CAM visualization
# Inside the right_col section, update the predict_button code:

# RIGHT COLUMN - X-ray Prediction
with right_col:
    st.subheader(get_text("📸 X-ray Prediction", "📸 Dự đoán X-quang"))

    uploaded_file = st.file_uploader(
        get_text("Choose X-ray (JPG/PNG)", "Chọn ảnh X-quang (JPG/PNG)"),
        type=["jpg", "png"]
    )

    if uploaded_file is not None:
        st.success(get_text("✅ Image uploaded successfully!", "✅ Tải ảnh lên thành công!"))

        # Add a toggle to show/hide the image
        show_image = st.checkbox(get_text("Show Image", "Hiển thị ảnh"), value=True)

        # Only display the image if the toggle is on
        if show_image:
            st.image(uploaded_file, width=700)

        # Add options for Grad-CAM layers
        grad_cam_layer = st.selectbox(
            get_text("Select layer for Grad-CAM visualization:", "Chọn lớp cho trực quan hóa Grad-CAM:"),
            ["conv1", "conv2", "conv3", "conv4", "conv5"]
        )

        # Add a checkbox to show/hide Grad-CAM
        show_gradcam = st.checkbox(get_text("Show Grad-CAM visualization", "Hiển thị trực quan hóa Grad-CAM"),
                                   value=True)

        predict_button = st.button(get_text("Predict", "Dự đoán"))

        if predict_button:
            try:
                # Image preprocessing
                image = Image.open(uploaded_file).convert("RGB")
                original_image = image.copy()  # Keep a copy of the original image

                # Get current device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Load model and predict
                model_lung = load_model()

                if model_lung is not None:
                    tensor = preprocess_image(image).to(device)

                    # Get prediction
                    with torch.no_grad():
                        output = model_lung(tensor)
                        prediction = torch.argmax(output, 1).item()

                        classes = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']
                        condition = classes[prediction]

                        result_prefix = "🧑🏽‍⚕️ 🙏🏽 **Prediction:**" if st.session_state.language == "en" else "🧑🏽‍⚕️ 🙏🏽 **Kết quả dự đoán:**"
                        result = f"{result_prefix} {condition}"
                        st.session_state.prediction_result = result
                        st.success(result)

                    # Generate and display Grad-CAM visualization if requested
                    if show_gradcam:
                        with st.spinner(
                                get_text("Generating Grad-CAM visualization...", "Đang tạo trực quan hóa Grad-CAM...")):
                            # Get the target layer based on user selection
                            target_layer_map = {
                                "conv1": model_lung.conv1,
                                "conv2": model_lung.conv2,
                                "conv3": model_lung.conv3,
                                "conv4": model_lung.conv4,
                                "conv5": model_lung.conv5
                            }
                            target_layer = target_layer_map[grad_cam_layer]

                            # Generate Grad-CAM visualization
                            superimposed_img, heatmap, _ = generate_gradcam_image(
                                model_lung, tensor, target_layer, original_image
                            )

                            # Display the Grad-CAM visualization
                            st.markdown(get_text("### Grad-CAM Visualization", "### Trực quan hóa Grad-CAM"))
                            st.image(
                                superimposed_img,
                                caption=get_text(f"Areas of focus for {condition} prediction",
                                                 f"Vùng tập trung cho dự đoán {condition}"),
                                width=700  # Use a specific width instead
                            )

                            st.markdown(get_text(
                                "The highlighted areas show which parts of the X-ray were most important for the model's prediction.",
                                "Các vùng được đánh dấu cho thấy những phần nào của X-quang quan trọng nhất cho dự đoán của mô hình."
                            ))

                        # Get treatment recommendations for the condition
                        if condition != "Normal":
                            st.info(
                                get_text("Generating treatment recommendations...", "Đang tạo khuyến nghị điều trị..."))
                            treatment_info = get_treatment_recommendations(gemini_model, condition,
                                                                           st.session_state.language)
                            st.markdown(get_text("### Treatment Recommendations", "### Khuyến nghị điều trị"))
                            st.markdown(treatment_info)
                        else:
                            st.markdown(get_text("### Normal X-ray", "### X-quang bình thường"))
                            normal_text = get_text(
                                "No specific treatment needed as the X-ray appears normal. Continue with regular health practices and consult with a healthcare provider for any persistent symptoms.",
                                "Không cần điều trị cụ thể vì X-quang có vẻ bình thường. Tiếp tục thực hành sức khỏe thông thường và tham khảo ý kiến của nhà cung cấp dịch vụ chăm sóc sức khỏe nếu có bất kỳ triệu chứng dai dẳng nào."
                            )
                            st.markdown(normal_text)

                else:
                    error_msg = "Cannot load model. Please, check your path." if st.session_state.language == "en" else "Không thể tải mô hình. Vui lòng kiểm tra đường dẫn của bạn."
                    st.error(error_msg)
            except Exception as e:
                error_prefix = "Error:" if st.session_state.language == "en" else "Lỗi:"
                st.error(f"{error_prefix} {str(e)}")
