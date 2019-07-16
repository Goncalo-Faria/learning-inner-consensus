## How to use

#Download mnist dataset
```
!wget https://storage.googleapis.com/capsule_toronto/mnist_data.tar.gz
!tar -zxvf mnist_data.tar.gz
!rm mnist_data.tar.gz
```
#Download cifar10 dataset
```
!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -zxvf cifar-10-binary.tar.gz
!rm cifar-10-binary.tar.gz
```
#Example of execution in python script

```
import sys
import subprocess

subprocess.call([sys.executable,    
                 'experiment.py',
                 "--dataset", "mnist",
                 "--train", "True",
                 "--data_dir", "mnist_data/",
                 "--model", "CapsuleBaseline",
                 "--batch_size", "32",
                 "--num_saves", "100",
                 "--max_steps", "10",
                 "--learning_rate", "0.001",
                 "--summary_dir", "tf_log"])
```

# Full list of arguments
```
('--data_dir', default=None, help='The data directory.',type=str)
('--eval_size', default=10000, help='Size of the test dataset.', type=int)
('--learning_rate', default=0.001,help='Size of the test dataset.', type=float)
('--batch_size', default=16, help='Batch size.', type=int)
('--max_steps', default=1000,help='Number of steps to train.', type=int)
('--model', default='capsule',help='The model to use for the experiment. capsule or baseline', type=str)
('--dataset', default='mnist', type=str,help='The dataset to use for the experiment.mnist, norb, cifar10.')
('--num_gpus', default=1,type=int,help='Number of gpus to use.')
('--num_targets', default=1,type=int,help='Number of targets to detect (1 or 2).')
('--regulizer_constant',default=0.0,type=float,help='scale of the sum of the regularizes.')
('--num_trials', default=1,type=int, help='Number of trials for ensemble evaluation.')
('--num_saves', default=100,type=int,help='number of checkpoints.')
('--show_step',default=5,type=int,help='How often to print.')
('--summary_dir', default="",type=str, help='Main directory for the experiments.')
('--checkpoint', default=None,type=str, help='The model checkpoint for evaluation.')
('--remake', default=False,type=bool,help='use reconstruction as regulizer.')
('--train', default=True,type=bool,help='Either train the model or test the model.')
('--validate', default=False,type=bool,help='Run trianing/eval in validation mode.')
('--budget_threshold', default=0.9,type=float,help='model saving threshold')
('--num_classes',default=10,type=int,help='number of classes in the dataset.')
('--verbose', default=True,type=bool, help='Register model info.')
('--loss_type', default='softmax',type=str,help=' classfication head. ')
```