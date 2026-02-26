0.Function Overview
This is On Going project.
You can see a small sample python code for communication between LLM and "Rule Base PSI".

For more information SEE related article on following URL (Japanese) 
https://note.com/osuosu1123/n/n4d34ccbb594f

1. STARTER is this
In python interpretor env.

1) You can see 4 steps Plan_Update_Operation in queue.
YOUR_REPO> python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error

2) Tou can see 4 steps log applied.
python -m tools.show_index --index_path runs\one_node\_index.jsonl --n 5 --verify
python -m tools.show_index --index_path runs\one_node\_index.jsonl --n 20 --json --verify 

3) Top routine for cooperation between this Rule Base Planner and a LLM web service with MANUAL INTERRAPTION. 
YOUR_REPO> python -m tools.run_cycle

Using run: runs\one_node\20260215_225917  # <<== TIME STAMP generated

================================
Diagnostic Package Created
Location: reports\20260223_080013 # <<== Open TIME STAMP generated
Next:
1. Open PROMPT_TO_LLM.md
2. Upload PNG to ChatGPT/Gemini
3. Save JSON as candidate_operator_spec.json
================================
