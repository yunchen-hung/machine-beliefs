# Cell 1: Setup and Installations
# !pip install transformers accelerate pillow kagglehub

# Cell 2: Imports & Configuration
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
import kagglehub
import glob
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Cell 3: Data Loading (Consistent with other notebooks)
print("Downloading/Loading FG-NET dataset...")
path = kagglehub.dataset_download("aiolapo/fgnet-dataset")


def get_image_path(pattern):
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No match found for pattern: {pattern}")
    return matches[0]


age_paths = {
    'newborn': get_image_path(f"{path}/**/080A00.JPG"),
    'older_infant': get_image_path(f"{path}/**/080A01.JPG"),
    'toddler': get_image_path(f"{path}/**/080A02.JPG"),
    'preschool_child': get_image_path(f"{path}/**/080A04.JPG"),
    'schoolage_child': get_image_path(f"{path}/**/080A07.JPG")
}

# Cell 4: Define Questions
questions_on_empiricism = [
    'Alex can see things with his eyes. When could Alex see with his eyes for the first time?',
    'When there is a sound close by, Alex can hear it. When could Alex hear sounds for the first time?',
    'When seeing a red flower and a blue flower, Alex can tell that they are different colors. Alex can tell colors apart. When could Alex tell colors apart for the first time?',
    'When there is a car approaching, Alex can tell that the car is getting closer. Alex can tell what is near and what is far. When could Alex tell near and far for the first time?',
    'When Alex sees someone hold an object and then drop it, Alex thinks the object will fall. Alex thinks objects will fall if we let go of them. When could Alex think that for the first time?',
    'If Alex sees a toy being hidden in a box, he will think the object is still there even though he can no longer see it. When could Alex think that for the first time?',
    'If Alex sees two cookies, one with 5 chocolate chips in it and one with 20 chocolate chips in it, he can tell which cookie has more chocolate chips without counting. When could Alex tell which has more for the first time?',
    'If Alex sees a turtle that is upside down and struggling to get on its feet, he thinks that he should help the turtle. Alex thinks that helping is the right thing to do. When could Alex think that for the first time?',
    'Alex can read books. When could Alex read for the first time?'
]

# Cell 5: Model Initialization
print("Loading Idefics2 model...")
model_id = "HuggingFaceM4/idefics2-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2" if device == "cuda" else "eager"
).to(device)

print("Model loaded successfully.")

# Cell 6: The Experiment Loop
print(f"Starting Inference on {len(questions_on_empiricism)} questions...")
print("=" * 60)

# Load images as PIL objects
image1 = load_image(age_paths['newborn'])
image2 = load_image(age_paths['toddler'])
image3 = load_image(age_paths['schoolage_child'])

results = []

for i, question in enumerate(questions_on_empiricism):
    # Idefics2 uses a specific chat template for interleaved images and text
    # We construct a prompt that presents the images first, then the question
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Image 1 (Newborn)."},
                {"type": "image"},
                {"type": "text", "text": "Image 2 (Toddler)."},
                {"type": "image"},
                {"type": "text", "text": "Image 3 (School-age child)."},
                {"type": "text",
                 "text": f"\n\nQuestion: {question}\nTask: Pick the image number (1, 2, or 3) that best answers the question. Reply with just the image number."}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process inputs
    inputs = processor(text=[text], images=[image1, image2, image3], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Extract only the assistant's response (remove the prompt)
    # Note: Idefics2 output includes the input prompt, so we split to get the new text
    # The prompt ends with "Assistant:" usually added by the template
    full_response = generated_texts[0]
    # Simple parsing to get the last part
    if "Assistant:" in full_response:
        answer = full_response.split("Assistant:")[-1].strip()
    else:
        # Fallback if template differs slightly, though apply_chat_template usually handles this
        # We can also just take the last few words if needed, but let's try to parse the prompt out
        # Since we know the prompt structure, we can verify what was added.
        # For now, let's print the full response to debug, and store a cleaned version.
        answer = full_response.split(messages[0]['content'][-1]['text'])[-1].strip()

    results.append(answer)
    print(f"Q{i + 1}: {question}")
    print(f"Idefics2 Answer: {answer}")
    print("-" * 60)

# Cell 7: Save/Export Data
print("Raw Results for Plotting:")
print(results)