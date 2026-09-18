from datasets import get_dataset_config_names, load_dataset

# 1. Kiểm tra tất cả các subset/config có trong dataset
configs = get_dataset_config_names("LLM-Digital-Twin/Twin-2K-500")
print("Các subset có sẵn:", configs)

# 2. Tải metadata hoặc dictionary để xem danh sách câu hỏi gốc (question text)
# Codebook sẽ liệt kê chi tiết từng câu hỏi nhỏ cấu thành nên 256 biến đó
print('data config[0]')
dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", configs[0])
print(dataset)

print('data config[1]')
dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", configs[1])
print(dataset)


dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", name="wave_split")

# Dữ liệu nằm trong split 'data' (hoặc 'train' tùy thuộc vào cách HF định nghĩa)
data = dataset['data']

# 2. Lấy người đầu tiên (hàng chỉ số 0)
first_person = data[0]

print(f"User ID (pid): {first_person['pid']}")
print("=" * 50)
print("Wave 1-3 Persona Text (Dữ liệu lịch sử):")
print("=" * 50)
print(first_person['wave1_3_persona_text'])