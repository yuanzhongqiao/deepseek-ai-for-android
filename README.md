<div class="Box-sc-g0xbh4-0 QkQOb js-snippet-clipboard-copy-unpositioned undefined" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto" _msttexthash="19498310" _msthash="185">deepseek-ai-安卓版</h1><a id="user-content-deepseek-ai-for-android" class="anchor" aria-label="永久链接：deepseek-ai-for-android" href="#deepseek-ai-for-android" _mstaria-label="962468" _msthash="186"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="231511904" _msthash="187">下面是一个详细的 <strong _istranslated="1">README.md</strong> 文件格式，用于 GitHub 记录您的教程：</p>
<div class="highlight highlight-text-md notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-mh"># <span class="pl-en">How to Use DeepSeek AI Model in Android Apps 🚀</span></span>

This repository provides a <span class="pl-s">**</span>step-by-step guide<span class="pl-s">**</span> on how to use the <span class="pl-s">**</span>DeepSeek AI<span class="pl-s">**</span> model in Android apps. Although the model size is quite large for mobile devices (the smallest model is around 4GB), it's still an exciting experiment worth trying!

<span class="pl-mh">### <span class="pl-en">Model Used:</span></span>
👉 <span class="pl-s">[</span>DeepSeek-R1-Distill-Qwen-1.5B<span class="pl-s">]</span><span class="pl-s">(</span><span class="pl-corl">https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</span><span class="pl-s">)</span>  

<span class="pl-mh">## <span class="pl-en">Overview 📖</span></span>
<span class="pl-s">1</span><span class="pl-v">.</span> <span class="pl-s">**</span>Download the DeepSeek Model<span class="pl-s">**</span> from Hugging Face.
<span class="pl-s">2</span><span class="pl-v">.</span> <span class="pl-s">**</span>Convert the Model<span class="pl-s">**</span> into ONNX, TensorFlow, and TensorFlow Lite formats.
<span class="pl-s">3</span><span class="pl-v">.</span> <span class="pl-s">**</span>Implement a Chat Interface<span class="pl-s">**</span> in an Android app using Jetpack Compose.

<span class="pl-ms">---</span>

<span class="pl-mh">## <span class="pl-en">Step 1: Download the DeepSeek Model 🧩</span></span>
<span class="pl-s">1</span><span class="pl-v">.</span> Visit the model page on Hugging Face: <span class="pl-s">[</span>DeepSeek-R1-Distill-Qwen-1.5B<span class="pl-s">]</span><span class="pl-s">(</span><span class="pl-corl">https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</span><span class="pl-s">)</span>.
<span class="pl-s">2</span><span class="pl-v">.</span> Clone the repository or download the model files directly:
   <span class="pl-s">```</span><span class="pl-en">bash</span>
   git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# How to Use DeepSeek AI Model in Android Apps 🚀

This repository provides a **step-by-step guide** on how to use the **DeepSeek AI** model in Android apps. Although the model size is quite large for mobile devices (the smallest model is around 4GB), it's still an exciting experiment worth trying!

### Model Used:
👉 [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)  

## Overview 📖
1. **Download the DeepSeek Model** from Hugging Face.
2. **Convert the Model** into ONNX, TensorFlow, and TensorFlow Lite formats.
3. **Implement a Chat Interface** in an Android app using Jetpack Compose.

---

## Step 1: Download the DeepSeek Model 🧩
1. Visit the model page on Hugging Face: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
2. Clone the repository or download the model files directly:
   ```bash
   git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<hr>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="90311702" _msthash="188">第 2 步：将模型转换为 ONNX 格式 🔄</h2><a id="user-content-step-2-convert-the-model-to-onnx-format-" class="anchor" aria-label="永久链接：第 2 步：将模型转换为 ONNX 格式 🔄" href="#step-2-convert-the-model-to-onnx-format-" _mstaria-label="75040225" _msthash="189"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="269301838" _msthash="190">ONNX 允许模型与不同的框架兼容。使用以下 Python 脚本将 Hugging Face 模型转换为 ONNX：</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">from</span> <span class="pl-s1">transformers</span> <span class="pl-k">import</span> <span class="pl-v">AutoModelForCausalLM</span>, <span class="pl-v">AutoTokenizer</span>
<span class="pl-k">import</span> <span class="pl-s1">torch</span>
<span class="pl-k">from</span> <span class="pl-s1">onnxruntime</span>.<span class="pl-s1">transformers</span> <span class="pl-k">import</span> <span class="pl-s1">optimizer</span>

<span class="pl-c"># Load the DeepSeek model and tokenizer</span>
<span class="pl-s1">model_name</span> <span class="pl-c1">=</span> <span class="pl-s">"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"</span>
<span class="pl-s1">model</span> <span class="pl-c1">=</span> <span class="pl-v">AutoModelForCausalLM</span>.<span class="pl-c1">from_pretrained</span>(<span class="pl-s1">model_name</span>)
<span class="pl-s1">tokenizer</span> <span class="pl-c1">=</span> <span class="pl-v">AutoTokenizer</span>.<span class="pl-c1">from_pretrained</span>(<span class="pl-s1">model_name</span>)

<span class="pl-c"># Convert to ONNX</span>
<span class="pl-s1">dummy_input</span> <span class="pl-c1">=</span> <span class="pl-en">tokenizer</span>(<span class="pl-s">"Hello, how are you?"</span>, <span class="pl-s1">return_tensors</span><span class="pl-c1">=</span><span class="pl-s">"pt"</span>)
<span class="pl-s1">torch</span>.<span class="pl-c1">onnx</span>.<span class="pl-c1">export</span>(
    <span class="pl-s1">model</span>,
    (<span class="pl-s1">dummy_input</span>[<span class="pl-s">"input_ids"</span>],),
    <span class="pl-s">"deepseek_model.onnx"</span>,
    <span class="pl-s1">input_names</span><span class="pl-c1">=</span>[<span class="pl-s">"input_ids"</span>],
    <span class="pl-s1">output_names</span><span class="pl-c1">=</span>[<span class="pl-s">"output"</span>],
    <span class="pl-s1">dynamic_axes</span><span class="pl-c1">=</span>{<span class="pl-s">"input_ids"</span>: {<span class="pl-c1">0</span>: <span class="pl-s">"batch_size"</span>, <span class="pl-c1">1</span>: <span class="pl-s">"sequence_length"</span>}}
)</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from onnxruntime.transformers import optimizer

# Load the DeepSeek model and tokenizer
model_name = &quot;deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B&quot;
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convert to ONNX
dummy_input = tokenizer(&quot;Hello, how are you?&quot;, return_tensors=&quot;pt&quot;)
torch.onnx.export(
    model,
    (dummy_input[&quot;input_ids&quot;],),
    &quot;deepseek_model.onnx&quot;,
    input_names=[&quot;input_ids&quot;],
    output_names=[&quot;output&quot;],
    dynamic_axes={&quot;input_ids&quot;: {0: &quot;batch_size&quot;, 1: &quot;sequence_length&quot;}}
)" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<hr>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="101116457" _msthash="191">第 3 步：将 ONNX 转换为 TensorFlow 模型 🔧</h2><a id="user-content-step-3-convert-onnx-to-tensorflow-model-" class="anchor" aria-label="永久链接：第 3 步：将 ONNX 转换为 TensorFlow 模型 🔧" href="#step-3-convert-onnx-to-tensorflow-model-" _mstaria-label="76596923" _msthash="192"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font _mstmutation="1" _msttexthash="95661397" _msthash="193">使用该库将 ONNX 模型转换为 TensorFlow 格式：</font><code>onnx-tf</code></p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>pip install onnx-tf</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="pip install onnx-tf" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto" _msttexthash="42850665" _msthash="194">然后运行以下脚本：</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>from onnx_tf.backend import prepare
import onnx

<span class="pl-c"><span class="pl-c">#</span> Load the ONNX model</span>
onnx_model = onnx.load(<span class="pl-s"><span class="pl-pds">"</span>deepseek_model.onnx<span class="pl-pds">"</span></span>)

<span class="pl-c"><span class="pl-c">#</span> Convert to TensorFlow</span>
tf_rep = prepare(onnx_model)
tf_rep.export_graph(<span class="pl-s"><span class="pl-pds">"</span>deepseek_model_tf<span class="pl-pds">"</span></span>)</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="from onnx_tf.backend import prepare
import onnx

# Load the ONNX model
onnx_model = onnx.load(&quot;deepseek_model.onnx&quot;)

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph(&quot;deepseek_model_tf&quot;)" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<hr>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="120403335" _msthash="195">第 4 步：将 TensorFlow 模型转换为 TensorFlow Lite 🪶</h2><a id="user-content-step-4-convert-tensorflow-model-to-tensorflow-lite-" class="anchor" aria-label="永久链接：第 4 步：将 TensorFlow 模型转换为 TensorFlow Lite 🪶" href="#step-4-convert-tensorflow-model-to-tensorflow-lite-" _mstaria-label="92258946" _msthash="196"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="208720980" _msthash="197">通过将模型转换为 TensorFlow Lite （TFLite） 模型来减小模型大小。</p>
<p dir="auto" _msttexthash="25591501" _msthash="198">安装 TensorFlow Lite：</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>pip install tensorflow</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="pip install tensorflow" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto" _msttexthash="23618192" _msthash="199">运行脚本：</p>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">import</span> <span class="pl-s1">tensorflow</span> <span class="pl-k">as</span> <span class="pl-s1">tf</span>

<span class="pl-c"># Convert the TensorFlow model to TFLite</span>
<span class="pl-s1">converter</span> <span class="pl-c1">=</span> <span class="pl-s1">tf</span>.<span class="pl-c1">lite</span>.<span class="pl-c1">TFLiteConverter</span>.<span class="pl-c1">from_saved_model</span>(<span class="pl-s">"deepseek_model_tf"</span>)
<span class="pl-s1">tflite_model</span> <span class="pl-c1">=</span> <span class="pl-s1">converter</span>.<span class="pl-c1">convert</span>()

<span class="pl-c"># Save the TFLite model</span>
<span class="pl-k">with</span> <span class="pl-en">open</span>(<span class="pl-s">"deepseek_model.tflite"</span>, <span class="pl-s">"wb"</span>) <span class="pl-k">as</span> <span class="pl-s1">f</span>:
    <span class="pl-s1">f</span>.<span class="pl-c1">write</span>(<span class="pl-s1">tflite_model</span>)</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="import tensorflow as tf

# Convert the TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(&quot;deepseek_model_tf&quot;)
tflite_model = converter.convert()

# Save the TFLite model
with open(&quot;deepseek_model.tflite&quot;, &quot;wb&quot;) as f:
    f.write(tflite_model)" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<hr>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="131697891" _msthash="200">第 5 步：将 TFLite 模型集成到 Android 应用中 📱</h2><a id="user-content-step-5-integrate-the-tflite-model-into-android-app-" class="anchor" aria-label="永久链接：第 5 步：将 TFLite 模型集成到 Android 应用中 📱" href="#step-5-integrate-the-tflite-model-into-android-app-" _mstaria-label="90302277" _msthash="201"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="24972519" _msthash="202">添加 TFLite 依赖项</h3><a id="user-content-add-tflite-dependencies" class="anchor" aria-label="永久链接：添加 TFLite 依赖项" href="#add-tflite-dependencies" _mstaria-label="876070" _msthash="203"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font _mstmutation="1" _msttexthash="151479107" _msthash="204">将 TensorFlow Lite 依赖项添加到您的 Android 应用的 ：</font><code>build.gradle</code></p>
<div class="highlight highlight-source-groovy-gradle notranslate position-relative overflow-auto" dir="auto"><pre>implementation <span class="pl-s"><span class="pl-pds">'</span>org.tensorflow:tensorflow-lite:2.12.0<span class="pl-pds">'</span></span>
implementation <span class="pl-s"><span class="pl-pds">'</span>org.tensorflow:tensorflow-lite-support:0.4.0<span class="pl-pds">'</span></span></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="implementation 'org.tensorflow:tensorflow-lite:2.12.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.0'" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto" _msttexthash="37255270" _msthash="205">创建简单的聊天界面</h3><a id="user-content-create-a-simple-chat-interface" class="anchor" aria-label="永久链接： 创建一个简单的聊天界面" href="#create-a-simple-chat-interface" _mstaria-label="1151202" _msthash="206"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="213449418" _msthash="207">使用 <strong _istranslated="1">Jetpack Compose</strong> 实现类似聊天的 UI。以下是完整的 Kotlin 代码：</p>
<div class="highlight highlight-source-kotlin notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-k">import</span> <span class="pl-smi">android.os.Bundle</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.activity.ComponentActivity</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.activity.compose.setContent</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.compose.foundation.layout.*</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.compose.foundation.text.BasicTextField</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.compose.material3.*</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.compose.runtime.*</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.compose.ui.Modifier</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.compose.ui.unit.dp</span>
<span class="pl-k">import</span> <span class="pl-smi">androidx.lifecycle.lifecycleScope</span>
<span class="pl-k">import</span> <span class="pl-smi">kotlinx.coroutines.Dispatchers</span>
<span class="pl-k">import</span> <span class="pl-smi">kotlinx.coroutines.launch</span>

<span class="pl-k">class</span> <span class="pl-en">MainActivity</span> : <span class="pl-en">ComponentActivity</span>() {
    <span class="pl-k">override</span> <span class="pl-k">fun</span> <span class="pl-en">onCreate</span>(<span class="pl-smi">savedInstanceState</span><span class="pl-k">:</span> <span class="pl-en">Bundle</span><span class="pl-k">?</span>) {
        <span class="pl-c1">super</span>.onCreate(savedInstanceState)
        setContent {
            <span class="pl-en">ChatScreen</span>()
        }
    }
}

@Composable
<span class="pl-k">fun</span> <span class="pl-en">ChatScreen</span>() {
    <span class="pl-k">var</span> userInput by remember { mutableStateOf(<span class="pl-s"><span class="pl-pds">"</span><span class="pl-pds">"</span></span>) }
    <span class="pl-k">var</span> response by remember { mutableStateOf(<span class="pl-s"><span class="pl-pds">"</span>Hi! How can I help you?<span class="pl-pds">"</span></span>) }

    <span class="pl-en">Column</span>(
        modifier <span class="pl-k">=</span> <span class="pl-en">Modifier</span>.fillMaxSize().padding(<span class="pl-c1">16</span>.dp),
        verticalArrangement <span class="pl-k">=</span> <span class="pl-en">Arrangement</span>.<span class="pl-en">SpaceBetween</span>
    ) {
        <span class="pl-en">Text</span>(
            text <span class="pl-k">=</span> <span class="pl-s"><span class="pl-pds">"</span>Chat with DeepSeek AI<span class="pl-pds">"</span></span>,
            style <span class="pl-k">=</span> <span class="pl-en">MaterialTheme</span>.typography.headlineMedium,
            modifier <span class="pl-k">=</span> <span class="pl-en">Modifier</span>.padding(bottom <span class="pl-k">=</span> <span class="pl-c1">8</span>.dp)
        )
        <span class="pl-en">Column</span>(
            modifier <span class="pl-k">=</span> <span class="pl-en">Modifier</span>.weight(<span class="pl-c1">1f</span>).padding(bottom <span class="pl-k">=</span> <span class="pl-c1">16</span>.dp)
        ) {
            <span class="pl-en">Text</span>(text <span class="pl-k">=</span> response, modifier <span class="pl-k">=</span> <span class="pl-en">Modifier</span>.fillMaxWidth().padding(<span class="pl-c1">8</span>.dp))
        }
        <span class="pl-en">BasicTextField</span>(
            value <span class="pl-k">=</span> userInput,
            onValueChange <span class="pl-k">=</span> { userInput <span class="pl-k">=</span> it },
            modifier <span class="pl-k">=</span> <span class="pl-en">Modifier</span>.fillMaxWidth().padding(<span class="pl-c1">8</span>.dp)
        )
        <span class="pl-en">Button</span>(
            onClick <span class="pl-k">=</span> {
                <span class="pl-c"><span class="pl-c">//</span> Call the TFLite model for a response</span>
                response <span class="pl-k">=</span> <span class="pl-s"><span class="pl-pds">"</span>Model processing: <span class="pl-e">$userInput</span><span class="pl-pds">"</span></span>
            },
            modifier <span class="pl-k">=</span> <span class="pl-en">Modifier</span>.fillMaxWidth()
        ) {
            <span class="pl-en">Text</span>(text <span class="pl-k">=</span> <span class="pl-s"><span class="pl-pds">"</span>Send<span class="pl-pds">"</span></span>)
        }
    }
}</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ChatScreen()
        }
    }
}

@Composable
fun ChatScreen() {
    var userInput by remember { mutableStateOf(&quot;&quot;) }
    var response by remember { mutableStateOf(&quot;Hi! How can I help you?&quot;) }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = &quot;Chat with DeepSeek AI&quot;,
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        Column(
            modifier = Modifier.weight(1f).padding(bottom = 16.dp)
        ) {
            Text(text = response, modifier = Modifier.fillMaxWidth().padding(8.dp))
        }
        BasicTextField(
            value = userInput,
            onValueChange = { userInput = it },
            modifier = Modifier.fillMaxWidth().padding(8.dp)
        )
        Button(
            onClick = {
                // Call the TFLite model for a response
                response = &quot;Model processing: $userInput&quot;
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = &quot;Send&quot;)
        }
    }
}" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<hr>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="23679097" _msthash="208">流程图 📊</h2><a id="user-content-flow-diagram-" class="anchor" aria-label="永久链接：流程图 📊" href="#flow-diagram-" _mstaria-label="41922374" _msthash="209"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="76855506" _msthash="210">以下是完整过程的简单流程图：</p>
<section class="js-render-needs-enrichment render-needs-enrichment position-relative" data-identity="9953b907-4491-49a4-968c-1b81ab733d55" data-host="https://viewscreen.githubusercontent.com" data-src="https://viewscreen.githubusercontent.com/markdown/mermaid?docs_host=https%3A%2F%2Fdocs.github.com" data-type="mermaid" aria-label="Mermaid 渲染输出容器" _mstaria-label="906295" _msthash="211">
  <div class="js-render-enrichment-target" data-json="{&quot;data&quot;:&quot;graph TD\nA[Download DeepSeek Model] --&amp;gt; B[Convert to ONNX]\nB --&amp;gt; C[Convert to TensorFlow]\nC --&amp;gt; D[Convert to TensorFlow Lite]\nD --&amp;gt; E[Integrate into Android App]\nE --&amp;gt; F[Chat Interface for Interaction]\n&quot;}" data-plain="graph TD
A[Download DeepSeek Model] --> B[Convert to ONNX]
B --> C[Convert to TensorFlow]
C --> D[Convert to TensorFlow Lite]
D --> E[Integrate into Android App]
E --> F[Chat Interface for Interaction]
" dir="auto"><!----><!----><div class="position-absolute top-0 pr-2 right-0 d-flex flex-justify-end flex-items-center">
    
    <details class="details-reset details-overlay details-overlay-dark" style="display: contents">
      <summary role="button" aria-label="“打开”对话框" class="btn my-2 mr-2 p-0 d-inline-flex" aria-haspopup="dialog" _mstaria-label="154752" _msthash="212">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" class="octicon m-2">
          <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 1.06L2.56 7h10.88l-2.22-2.22a.75.75 0 011.06-1.06l3.5 3.5a.75.75 0 010 1.06l-3.5 3.5a.75.75 0 11-1.06-1.06l2.22-2.22H2.56l2.22 2.22a.75.75 0 11-1.06 1.06l-3.5-3.5a.75.75 0 010-1.06l3.5-3.5z"></path>
        </svg>
      </summary>
      <details-dialog class="Box Box--overlay render-full-screen d-flex flex-column anim-fade-in fast" aria-label="mermaid rendered container" role="dialog" aria-modal="true" _msthidden="1" _mstaria-label="611598" _msthash="213">
        <div _msthidden="1">
          <button aria-label="Close dialog" data-close-dialog="" type="button" data-view-component="true" class="Link--muted btn-link position-absolute render-full-screen-close" _msthidden="A" _mstaria-label="177619" _msthash="214">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:text-bottom" class="octicon octicon-x">
              <path fill-rule="evenodd" d="M5.72 5.72a.75.75 0 011.06 0L12 10.94l5.22-5.22a.75.75 0 111.06 1.06L13.06 12l5.22 5.22a.75.75 0 11-1.06 1.06L12 13.06l-5.22 5.22a.75.75 0 01-1.06-1.06L10.94 12 5.72 6.78a.75.75 0 010-1.06z"></path>
            </svg>
          </button>
          <div class="Box-body border-0" role="presentation"></div>
        </div>
      </details-dialog>
    </details>
  <!----><clipboard-copy class="btn my-2 js-clipboard-copy p-0 d-inline-flex tooltipped-no-delay" role="button" data-copy-feedback="Copied!" data-tooltip-direction="s" aria-label="复制美人鱼代码" value="graph TD
A[Download DeepSeek Model] --> B[Convert to ONNX]
B --> C[Convert to TensorFlow]
C --> D[Convert to TensorFlow Lite]
D --> E[Integrate into Android App]
E --> F[Chat Interface for Interaction]
" tabindex="0" _mstaria-label="283933" _msthash="215">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" class="octicon octicon-copy js-clipboard-copy-icon m-2">
      <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path>
      <path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
    </svg>
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
      <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
    </svg>
  </clipboard-copy>
  </div><!---->
    <div class="render-container color-bg-transparent js-render-target p-0 is-render-automatic is-render-requested is-render-ready" data-identity="9953b907-4491-49a4-968c-1b81ab733d55" data-host="https://viewscreen.githubusercontent.com" data-type="mermaid" style="height: 694px;">
      <iframe title="File display" role="presentation" class="render-viewer" sandbox="allow-scripts allow-same-origin allow-top-navigation allow-popups" src="https://viewscreen.githubusercontent.com/markdown/mermaid?docs_host=https%3A%2F%2Fdocs.github.com&amp;color_mode=light#9953b907-4491-49a4-968c-1b81ab733d55" name="9953b907-4491-49a4-968c-1b81ab733d55" data-content="{&quot;data&quot;:&quot;graph TD\nA[Download DeepSeek Model] --&amp;gt; B[Convert to ONNX]\nB --&amp;gt; C[Convert to TensorFlow]\nC --&amp;gt; D[Convert to TensorFlow Lite]\nD --&amp;gt; E[Integrate into Android App]\nE --&amp;gt; F[Chat Interface for Interaction]\n&quot;}">
      </iframe>
    </div>
  <!----><!----></div>
  <span class="js-render-enrichment-loader d-flex flex-justify-center flex-items-center width-full" style="min-height:100px" role="presentation" hidden="" _msthidden="1">
    <span data-view-component="true" _msthidden="1">
  <svg style="box-sizing: content-box; color: var(--color-icon-primary);" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true" data-view-component="true" class="octospinner mx-auto anim-rotate">
    <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-opacity="0.25" stroke-width="2" vector-effect="non-scaling-stroke" fill="none"></circle>
    <path d="M15 8a7.002 7.002 0 00-7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke"></path>
</svg>    <span class="sr-only" _msttexthash="92391" _msthidden="1" _msthash="216">Loading</span>
</span>
  </span>
<div class="js-render-enrichment-fallback" _msthidden="1"><div class="render-plaintext-hidden" dir="auto" _msthidden="1">
      <pre lang="mermaid" aria-label="Raw mermaid code" _msthidden="A" _mstaria-label="254033" _msthash="217">graph TD
A[Download DeepSeek Model] --&gt; B[Convert to ONNX]
B --&gt; C[Convert to TensorFlow]
C --&gt; D[Convert to TensorFlow Lite]
D --&gt; E[Integrate into Android App]
E --&gt; F[Chat Interface for Interaction]
</pre>
    </div></div></section>

<hr>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto" _msttexthash="20590479" _msthash="218">结论 🎉</h2><a id="user-content-conclusion-" class="anchor" aria-label="永久链接：结论 🎉" href="#conclusion-" _mstaria-label="40703598" _msthash="219"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto" _msttexthash="1188655949" _msthash="220">DeepSeek AI 模型提供了一种将高级 AI 集成到 Android 应用程序中的强大方法。尽管其尺寸对移动设备来说具有挑战性，但它是迈向设备端智能未来的一步。🧠</p>
<p dir="auto" _msttexthash="214591429" _msthash="221">如有任何问题或疑问，请随时提出 GitHub 问题或在 <a href="https://www.linkedin.com/in/codewithpk/" rel="nofollow" _istranslated="1">LinkedIn</a> 上与我联系。</p>
<p dir="auto" _msttexthash="61229389" _msthash="222">访问我的网站上的更多教程 <a href="https://codewithpk.com/" rel="nofollow" _istranslated="1">CodeWithPK.com</a>。</p>
<hr>
<p dir="auto" _msttexthash="54795026" _msthash="223"><strong _istranslated="1">祝您编码愉快！</strong>💻✨</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>
Feel free to copy, customize, and upload it to your GitHub repository! 😊
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
</article></div>
