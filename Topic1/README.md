Step 4: Timings for the following setups:


MODEL_NAME = "Llama-3.2-1B-Instruct"
1. (Using GPU and no quantization.) - 0:49.361
2. (Using CPU and no quantization.) 2:15.12
3. (Using CPU and 4-bit quantization.) - 1:23:423

MODEL_NAME = "Qwen2.5-0.5B"
1. (Using GPU and no quantization.) - 53.761218s
2. (Using CPU and no quantization.) 138.236871
3. (Using CPU and 4-bit quantization.) - 101.54675s

MODEL_NAME = "Mistral-7B-Instruct-v0.3"
1. (Using GPU and no quantization.) - 5:12.847
2. (Using CPU and no quantization.) - 18:47.392
3. (Using CPU and 4-bit quantization.) - 10:34.615



Based on the graph and analysis, yes, there are clear patterns - the mistakes are NOT random. Both models perform almost identically (43.1% vs 43.8%) and struggle with the same types of subjects. They both find Abstract Algebra hardest (~24-33% accuracy) and Astronomy easiest (~50% accuracy). The errors systematically cluster around mathematical reasoning and subjects requiring deep domain knowledge. However, the current data only shows subject-level results, not individual questions. To know if they make mistakes on the exact same questions (rather than just the same subjects), you'd need to modify the evaluation script to save per-question results and calculate the overlap. But the strong evidence suggests they likely do make similar mistakes since they have nearly identical accuracy and fail/succeed on the same subject categories.
