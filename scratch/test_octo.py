import jax
import octo
from octo.model.octo_model import OctoModel

print("Loading Octo-Small...")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
print("Loaded successfully!")
print(model)
