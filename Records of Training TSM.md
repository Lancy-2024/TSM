---


---

<h2 id="modification-in-original-tsm-code">Modification in Original TSM Code</h2>
<p><a href="https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py">main.py</a><br>
1.Add a parameter <strong>split_name</strong> in train_loader and val_loader to provide correct path. The value of split_name should be train or val.</p>
<p>For train_loader, the path should be /mnt/home/msc/k400/images/train…<br>
For val_loader, the path should be /mnt/home/msc/k400/images/val…</p>
<pre><code>train_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
               new_length=data_length,
               modality=args.modality,
               image_tmpl=prefix,
               transform=torchvision.transforms.Compose([
                   train_augmentation,
                   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                   normalize,
               ]), dense_sample=args.dense_sample, split_name='train'),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True,
    drop_last=True)  # prevent something not % n_GPU
</code></pre>
<p><a href="https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/utils.py">util.py</a><br>
2.Use correct[:k].<strong>reshape</strong> instead of correct[:k].view to fix an error</p>
<pre><code>def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
</code></pre>
<h2 id="modification-in-our-tsm-code">Modification in Our TSM Code</h2>
<p><a href="https://github.com/SAILTECHTEAM/Python-TSM-Training/blob/main/main.py">main.py</a><br>
1.Revise the value of num_classes, because there are 400 classes.</p>
<pre><code>#num_classes = 3 
num_classes = 400
</code></pre>
<p>2.Add a parameter <strong>split_name</strong> in train_loader and val_loader.<br>
3.Delete unused code about variables <strong>train_class_counts</strong> and <strong>val_class_counts</strong>. They are only used in loss function ‘Focal’ and ‘CB’, while we set the loss function as ‘nll’ for the whole training process.<br>
If not, some errors will occur when running.</p>
<pre><code>   """Loss function setting"""
# define loss function (criterion)and optimizer
#train_class_counts = {0: 0, 1: 0, 2: 0}
#val_class_counts = {0: 0, 1: 0, 2: 0}
train_class_counts = {i: 0 for i in range(400)}
val_class_counts = {i: 0 for i in range(400)}
'''
#Iterate over the training data loader and count class occurrences
for inputs, labels in train_loader:
    for label in labels:
        train_class_counts[int(label)] += 1
# Iterate over the validation data loader and count class occurrences
for inputs, labels in val_loader:
    for label in labels:
        val_class_counts[int(label)] += 1
#Calculate and print class weights for the train and validation set
'''
print("\nClass Counts (Train and Validation):")
</code></pre>
<p><a href="https://github.com/SAILTECHTEAM/Python-TSM-Training/blob/main/ops/utils.py">util.py</a><br>
4.Modify function <strong>model_performance(output, target)</strong><br>
For dataset with 400 classes, we should initialize <strong>tp, fp and fn</strong> with 400 elements so that each class has a dedicated counter.<br>
Also, use <strong>range(400)</strong> instead of range(3)</p>
<pre><code> # Initialize counts for each class
    #tp = [0, 0, 0]
    #fp = [0, 0, 0]
    #fn = [0, 0, 0]
    tp = np.zeros(400, dtype=np.int32)
    fp = np.zeros(400, dtype=np.int32)
    fn = np.zeros(400, dtype=np.int32)
    ...
    # Calculate precision, recall, and F1 score for each class
    precision = []
    recall = []
    f1 = []
    #for i in range(3):  # number of class
    for i in range(400):  # number of class
        if tp[i] + fp[i] == 0:
            precision.append(0)
        else:
            precision.append(tp[i] / (tp[i] + fp[i]))
        if tp[i] + fn[i] == 0:
            recall.append(0)
        else:
            recall.append(tp[i] / (tp[i] + fn[i]))   
        if precision[i] + recall[i] == 0:
            f1.append(0)
        else:
            f1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
</code></pre>
<h2 id="training">Training</h2>
<p>Use the same command to train the original TSM and our TSM.</p>
<pre><code>python main.py kinetics RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.005 --wd 1e-4 --lr_steps 20 40 --epochs 50 --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
</code></pre>
<p>Explanation:<br>
1.We decrease batch size from 128 to 32 because of memory limit on our server, and decrease learning rate from 0.02 to 0.005 in proportion to batch size.</p>

