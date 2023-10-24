# FLAN-T5 XXL Demo
### Efficient AI: Empowering Large Language Models with Intel® Extension for PyTorch to Combat Carbon Emissions
This demo showcases how Intel® Extension for PyTorch can reduce carbon emissions while maintaining the performance of Large Language Models (LLMs). With just a few lines of code, you will see the improvements that can be achieved in both latency and power consumption. Watch our demo to learn more!

##### Optional: Install libraries. You can find the full list of libraries and the version used for this demo in requirements.txt.
"""

#%pip install gradio
#%pip install sentencepiece
#%pip install codecarbon
#%pip install prometheus_client

"""##### Import libraries and download the model:"""

import torch
import codecarbon
from codecarbon import EmissionsTracker
from transformers import T5Tokenizer, T5ForConditionalGeneration

import gradio as gr
import time
import intel_extension_for_pytorch as ipex

# Upload the model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

"""##### Create the chat interface:"""

model.eval()
#Create a version of the model that is optimized by Intel Extension for PyTorch
model_o = ipex.optimize(model, dtype=torch.bfloat16)

#Gradio demo
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            text1 = gr.Button(value="PyTorch* with mkldnn backend turned off")
        with gr.Column(scale=1, min_width=600):
            text2 = gr.Button(value="PyTorch* + Intel® Extension for PyTorch Optimizer + Auto Mixed Precision")
    with gr.Row():
        #Unoptimized performance
        with gr.Column(scale=1, min_width=600):
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])

            def respond(message, chat_history):
                try:
                    input_ids = tokenizer(message, return_tensors="pt").input_ids
                    with torch.backends.mkldnn.flags(enabled=False):
                        tracker = EmissionsTracker()
                        tracker.start()
                        time1=time.time()
                        with torch.no_grad():
                            outputs_u = model.generate(input_ids, max_new_tokens=200)
                        time2=time.time()
                        emissions_no_ipex: float = tracker.stop()
                        emissions_no_ipex=emissions_no_ipex*1000000 #convert to mg

                    latency_u=time2-time1
                    latency_msg="Latency: "+str(latency_u)+" s"
                    emission_msg="Emissions: "+str(emissions_no_ipex)+" mgCO2.eq"
                    bot_message = str(tokenizer.decode(outputs_u[0]))+"\n"+latency_msg+"\n"+emission_msg
                except Exception as e:
                    bot_message = e.__doc__
                chat_history.append((message, bot_message))
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])

        #Optimized performance (model_o)
        with gr.Column(scale=1, min_width=600):

            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])


            def respond(message, chat_history):
                try:
                    input_ids = tokenizer(message, return_tensors="pt").input_ids
                    tracker = EmissionsTracker()
                    tracker.start()
                    time3=time.time()
                    with torch.no_grad(), torch.cpu.amp.autocast():
                        outputs_o = model_o.generate(input_ids, max_new_tokens=200)
                    time4=time.time()
                    emissions_ipex: float = tracker.stop()
                    emissions_ipex=emissions_ipex*1000000 #convert to mg

                    latency_o=time4-time3
                    latency_msg="Latency: "+str(latency_o)+" s"
                    emission_msg="Emissions: "+str(emissions_ipex)+" mgCO₂eq"
                    bot_message = str(tokenizer.decode(outputs_o[0]))+"\n"+latency_msg+"\n"+emission_msg

                except Exception as e:
                    bot_message = e.__doc__

                chat_history.append((message, bot_message))
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])

"""##### Launch the demo:"""

#share=True if you want a public link to get access to your demo
demo.launch(share=True)

