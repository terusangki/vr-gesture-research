import os
import jax
import jax.numpy as jnp
import numpy as np
import mediapy as media
from videoprism import models as vp


# 2. 모델 및 토크나이저 설정
# 공식 데모에서 사용하는 LVT(Language-Video-Transformer) 베이스 모델을 사용합니다.
MODEL_NAME = 'videoprism_lvt_public_v1_base'
NUM_FRAMES = 16
FRAME_SIZE = 288

print(f"Loading model: {MODEL_NAME}...")
flax_model = vp.get_model(MODEL_NAME)
loaded_state = vp.load_pretrained_weights(MODEL_NAME)
text_tokenizer = vp.load_text_tokenizer('c4_en')

# 3. 분석 대상 동작 클래스 정의 (교수님 요청 사항)
class_names = [
    "top-left tapping", "top tapping", "top-right tapping",
    "left tapping", "right tapping",
    "bottom-left tapping", "bottom tapping", "bottom-right tapping"
]
PROMPT_TEMPLATE = 'a video of {}.'
text_queries = [PROMPT_TEMPLATE.format(t) for t in class_names]

# 텍스트 데이터 토큰화
text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)

@jax.jit
def forward_fn(inputs, t_ids, t_paddings):
    """비디오와 텍스트 임베딩을 동시에 추출하는 순전파 함수"""
    return flax_model.apply(loaded_state, inputs, t_ids, t_paddings, train=False)

def preprocess_video(filename):
    """비디오 전처리 및 16프레임 샘플링"""
    frames = media.read_video(filename)
    frame_indices = np.linspace(0, len(frames), num=NUM_FRAMES, endpoint=False, dtype=np.int32)
    frames = np.array([frames[i] for i in frame_indices])
    frames = media.resize_video(frames, shape=(FRAME_SIZE, FRAME_SIZE))
    frames = media.to_float01(frames)
    return jnp.asarray(frames[None, ...])

def compute_similarity(v_emb, t_emb):
    """비디오와 텍스트 임베딩 간의 유사도 계산 (공식 데모 로직 적용)"""
    emb_dim = v_emb[0].shape[-1]
    v_emb = np.array(v_emb).reshape(-1, emb_dim)
    t_emb = np.array(t_emb).reshape(-1, emb_dim)
    
    # 내적(Dot product) 기반 유사도 산출 및 Softmax(temperature=0.01) 적용
    similarity = np.dot(v_emb, t_emb.T)
    similarity /= 0.01
    exp_sim = np.exp(similarity)
    return exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

# 4. 비디오 파일 분석 실행
video_dir = '/home/work/Zoogavin/'
video_files = [f for f in sorted(os.listdir(video_dir)) if f.endswith('.mp4')]

print("\n" + "="*60)
print("  VideoPrism Zero-shot Gesture Recognition Results")
print("="*60)

for video_name in video_files:
    video_path = os.path.join(video_dir, video_name)
    
    try:
        # 비디오 로드 및 전처리
        video_input = preprocess_video(video_path)
        
        # 모델 추론 (비디오 및 텍스트 특징 동시 추출)
        v_embeddings, t_embeddings, _ = forward_fn(video_input, text_ids, text_paddings)
        
        # 유사도 계산 및 결과 도출
        probs = compute_similarity(v_embeddings, t_embeddings)[0]
        best_idx = np.argmax(probs)
        
        print(f"[Video File]: {video_name}")
        print(f" - Recognized Gesture: {class_names[best_idx]}")
        print(f" - Similarity Score: {probs[best_idx]:.4f}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error during analysis of {video_name}: {e}")

print("\nAll tasks are completed.")
