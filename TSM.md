---


---

<h2 id="dataset-preparation"><strong>Dataset Preparation</strong></h2>
<p><strong>Kinetics 400 Dataset</strong></p>
<ol>
<li>Download videos and annotations (train.csv and val.csv) from <a href="https://github.com/cvdfoundation/kinetics-dataset">kinetics-dataset</a>.</li>
<li>Run <a href="https://github.com/cvdfoundation/kinetics-dataset/blob/main/arrange_by_classes.py">arrange_by_classes.py</a> to arrange videos by 400 classes.</li>
<li>Run <a href="https://github.com/mit-han-lab/temporal-shift-module/blob/master/tools/vid2img_kinetics.py">tools/vid2img_kinetics.py</a> to extract frames from videos.</li>
<li>Run <a href="https://github.com/mit-han-lab/temporal-shift-module/blob/master/tools/gen_label_kinetics.py">tools/gen_label_kinetics.py</a> to generate annotations (<strong>train_videofolder.txt</strong> and <strong>val_videofolder.txt</strong>)  needed for dataloader.</li>
</ol>
<p>train_videofolder.txt</p>
<blockquote>
<p>abseiling\-3B32lodo2M_000059_000069 250 0<br>
abseiling\-7kbO0v4hag_000107_000117 300 0<br>
abseiling\-Cv3NwxG_8g_000087_000097 300 0<br>
…<br>
item[0] = path, item[1] = frame number, item[2] = class id</p>
</blockquote>
<p>The directory of dataset is</p>
<blockquote>
<p>│   │   ├── ${DATASET}_train_list_videos.txt<br>
│   │   ├── ${DATASET}_val_list_videos.txt<br>
│   │   ├── annotations<br>
│   │   │   ├── train.csv<br>
│   │   │   ├── …<br>
│   │   ├── videos_train<br>
│   │   ├── videos_val<br>
│   │   │   ├── abseiling<br>
│   │   │   │   ├── 0wR5jVB-WPk_000417_000427.mp4<br>
│   │   │   │   ├── …<br>
│   │   │   ├── …<br>
│   │   │   ├── zumba<br>
│   │   ├── rawframes_train   #optional for image-based data<br>
│   │   ├── rawframes_val     #optional for image-based data</p>
</blockquote>
<p><strong>TSM Dataset</strong><br>
It is similar to Kinetics.<br>
The frame number is fixedly 24. Videos are not split train and val in advance.<br>
3 classes: normal, hitting, shaking</p>
<ol>
<li>Prepare videos and extract frames from videos.</li>
<li>Run <a href="https://github.com/SAILTECHTEAM/Python-TSM-Training/blob/main/tools/train_val_spilt.py">train_val_spilt.py</a> to get train.txt and val.txt, which are same as train_videofolder.txt.</li>
<li>Configure  <a href="https://github.com/SAILTECHTEAM/Python-TSM-Training/blob/main/ops/dataset_config.py">ops/dataset_config.py</a>  to add a method for loading TSM dataset.</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">return_ogcio_balance_upper</span><span class="token punctuation">(</span>modality<span class="token punctuation">)</span><span class="token punctuation">:</span>  
 filename_categories <span class="token operator">=</span> <span class="token string">'datasets_tsm/ogcio_balance/category.txt'</span>  
  <span class="token keyword">if</span> modality <span class="token operator">==</span> <span class="token string">'RGB'</span><span class="token punctuation">:</span>  
 prefix <span class="token operator">=</span> <span class="token string">'img_{:03d}.jpg'</span>  
  root_data <span class="token operator">=</span> <span class="token string">'datasets_tsm/ogcio_balance/images'</span>  
  filename_imglist_train <span class="token operator">=</span> <span class="token string">'datasets_tsm/ogcio_balance/train.txt'</span>  
  filename_imglist_val <span class="token operator">=</span> <span class="token string">'datasets_tsm/ogcio_balance/val.txt'</span>  
  <span class="token keyword">else</span><span class="token punctuation">:</span>  
  <span class="token keyword">raise</span> NotImplementedError<span class="token punctuation">(</span><span class="token string">'no such modality:'</span> <span class="token operator">+</span> modality<span class="token punctuation">)</span>  
  <span class="token keyword">return</span> filename_categories<span class="token punctuation">,</span> filename_imglist_train<span class="token punctuation">,</span> filename_imglist_val<span class="token punctuation">,</span> root_data<span class="token punctuation">,</span> prefix
</code></pre>
<p>The directory of dataset is</p>
<blockquote>
<p>│   │   ├── images<br>
│   │   │   ├── video1.mp4<br>
│   │   │   │   ├── img_001.jpg<br>
│   │   │   │   ├── …<br>
│   │   │   │   ├── img_024.jpg<br>
│   │   │   ├── …<br>
│   │   ├── train.txt<br>
│   │   ├── val.txt<br>
│   │   ├── catogeries.txt</p>
</blockquote>
<p>train.txt</p>
<blockquote>
<p>… 24 0</p>
</blockquote>
<h2 id="data-loader">Data Loader</h2>

