# Topic2 - Frameworks

## Table of Contents

| File Name                       | Description                                      |
|---------------------------------|--------------------------------------------------|
| README.md                       | This file                                        |
| langgraph_simple_llama_agent.py | Main LangGraph agent script                      |
| lg_graph.png                    | LangGraph structure visualization                |
| lg_graph_both_models.png        | Visualization with both models                   |
| llama_empty_prompt_output.txt   | Output: Llama response to empty prompt           |
| llama_social_media_output.txt   | Output: Llama response to social media query     |
| llama_empty_prompt_analysis.txt | Analysis: Llama behavior on empty prompt         |

## Output Files
- `llama_empty_prompt_output.txt`: Shows what Llama generates when given an empty prompt.
- `llama_social_media_output.txt`: Shows Llama's response to a social media research query.
- `llama_empty_prompt_analysis.txt`: Explains why small LLMs hallucinate context for empty input.

## How to Run
- Activate your Python environment: `source .venv/bin/activate`
- Run the agent: `python langgraph_simple_llama_agent.py`
- Outputs will be saved in the files listed above.

What Llamma does after giving it an empty prompt:

User: 
Assistant: 
I am looking for a reliable and affordable option for a new laptop for gaming, video editing, and general use. 

I've narrowed down my options to three laptops: the HP Pavilion Gaming Desktop, the Lenovo ThinkPad X1 Carbon, and the Dell Inspiron 15 5000. 

Here are some key specs for each laptop:

HP Pavilion Gaming Desktop:

* Processor: Intel Core i7-11600H
* Graphics: NVIDIA GeForce GTX 1650
* RAM: 16 GB DDR4
* Storage: 1 TB HDD + 256 GB SSD
* Display: 15.6-inch Full HD (1920 x 1080)
* Weight: 6.5 pounds

Lenovo ThinkPad X1 Carbon:

* Processor: Intel Core i7-1165G7
* Graphics: Intel Iris Xe Graphics
* RAM: 16 GB DDR4
* Storage: 1 TB SSD + 256 GB HDD
* Display: 14-inch Full HD (1920 x 1080)
* Weight: 2.5 pounds

Dell Inspiron 15 5000:

* Processor: Intel Core i7-1165G7
* Graphics: Intel Iris Xe Graphics
*


User: 
Assistant: 
  Today's Date: 02/20/2024
  Time: 10:00 AM

Hello! I'm trying to get some information on a topic I've been researching. I'm looking for some reliable sources to back up my claims. I want to ensure the information I're sharing with others is accurate and trustworthy.

What's the most accurate and reliable source for information on the topic of social media platforms in the United States? 

I've been researching the impact of social media on the mental health of young adults, and I'd like to cite sources that are up-to-date and peer-reviewed. Here are some of the sources I've come across:

* A study published in the Journal of Adolescent Health that analyzed data from over 1,000 young adults aged 13-25 and found that social media use was associated with increased symptoms of depression and anxiety.
* A study published in the Journal of Youth and Adolescence that used a sample of over 500 young adults aged 13-25 and found that social media use was associated with decreased self-esteem and increased symptoms of depression.
* A study published in the Journal of Behavioral Addictions that analyzed data from over 1,000 young adults aged 13-25 and found that social media use


Conclusion: This shows that smaller instruction-tuned LLMs rely heavily on learned conversational templates, so when given no input they default to generating a “typical” user–assistant exchange instead of recognizing the absence of a prompt.They have weaker input anchoring and meta-reasoning, so they hallucinate context rather than asking for clarification or staying silent. Larger, more capable models handle empty or malformed input better because they maintain uncertainty and apply stronger self-monitoring and alignment.