import os

os.makedirs('experiments/phase_a', exist_ok=True)

with open('configs/octo_se3.yaml', 'r') as f:
    base_text = f.read()

configs_to_make = [
    ('phase_a_steps04.yaml', 4, 0.1),
    ('phase_a_steps20.yaml', 20, 0.1),
    ('phase_a_steps50.yaml', 50, 0.1),
    ('phase_a_scale001.yaml', 10, 0.01),
    ('phase_a_scale05.yaml', 10, 0.5),
    ('phase_a_scale10.yaml', 10, 1.0),
    ('phase_a_baseline.yaml', 10, 0.1),
]

for name, steps, scale in configs_to_make:
    text = base_text.replace('n_flow_steps_train: 10', f'n_flow_steps_train: {steps}')
    text = text.replace('n_flow_steps_eval: 10', f'n_flow_steps_eval: {steps}')
    text = text.replace('source_scale: 0.1', f'source_scale: {scale}')
    with open(f'experiments/phase_a/{name}', 'w') as f:
        f.write(text)
