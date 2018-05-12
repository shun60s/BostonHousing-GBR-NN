# �{�X�g���n�E�X�̃f�[�^���g�������z�u�[�X�e�B���O��A  

## �T�v  

�{�X�g���n�E�X�̃f�[�^(13��ނ̎w�W�ƏZ��i�̃f�[�^�j�̗\���������Ȃ�scikit learn�̎�������ɁA  
�e�����=����؂̍\����m�邱�ƂŌ��z�u�[�X�e�B���O��A(Gradient Boosting regression)�̓���𗝉�����B  
�܂��A�j���[�����l�b�g���[�N�ŗ\�������ꍇ�Ɣ�r����B
  
[github repository](https://github.com/shun60s/BostonHousing-GBR-NN)  

##  ���z�u�[�X�e�B���O��A(Gradient Boosting regression)�ŏZ��i��\�������ꍇ  
### ������@  
  
```
python GBR.py
```

### ����  

���}�́A�e�X�g�f�[�^�̗\���덷�̕��ςƁA�\������傫���O�ꂽ���[�X�g10�̃��X�g�A�e�w�W��(���ΓI�ȁj�d�v�x�ł���B
![sample1](docs/GBR1.png)  
  
�덷�̕��ϒl�͏������Ă��A�傫���O�����P�[�X�����݂��Ă���B  
�܂��A�j���[�����l�b�g���[�N�ƈႢ�A�Z��i��\�����邽�߂Ɏg�����w�W�i���́j�́i���ΓI�ȁj�d�v�x�̏���������B  

���Ȃ݂ɁA���֊֌W��T�邽�߁A���ꂼ��̎w�W�ƏZ��i�̒l���v���b�g����ƈȉ��̂悤�ɂȂ�B  
LSTAT: �Ꮚ���҂̊���  
![sample2](docs/LSTAT.png)  


RM: 1�˂�����̕��ϕ�����  
![sample3](docs/RM.png)  


DIS: �{�X�g���̎��5�̌ٗp���܂ł̏d�ݕt������  
![sample4](docs/DIS.png)  


INDUS: �񏬔��Ƃ̓y�n�ʐς̊����i�l���P�ʁj  
![sample5](docs/INDUS.png)  
  
�����܂ł���Ƒ��ւ͂悭�킩��Ȃ��B  
  


###�@�����(estimator)=�����(decision tree)�̍\��  

dot�t�H���_�̒��Ɉȉ�������B  

- tree0.dot,tree498.dot�Ȃǁ@�����(estimator)=�����(decision tree)�̍\���� dot�t�@�C���Ƃ��ď����o�����T���v��  
- tree0.png,tree498.png�Ȃǁ@�����(estimator)=�����(decision tree)�̍\���� png�i�}�`�O���t�j�ɕϊ������T���v��  
- dot2png.bat �i�ʂ̃\�t�g�́jdot.exe���g����dot�t�@�C����png�i�}�`�O���t�j�ɕϊ����邽�߂�windows�p�̃o�b�`�t�@�C��  
  

tree0 �͂P�Ԗڂ̐����(estimator)�Atree498 ��500�Ԗڂ̐����(estimator)�������B  
���Ԃ��オ���Ă����قǁA�[���̗t��value�l�͏������Ȃ��Ă����悤���B  
�@�@


���}�́A���낢��Ȏ�@�ɂ�镪�ނ̋��E���̗�ł��邪�A�����(decision tree)��Yes/No����Ȃ̂Łi�Ȑ��ł͂Ȃ��j���U�I�ȋ��E�ƂȂ�B  
![sample8](docs/border_line.png)  

  
## �j���[�����l�b�g���[�N���g���ďZ��i��\�������ꍇ  

### ������@  

�j���[�����l�b�g���[�N�̍\����4�w��FC�ŁA�t���[�����[�N�Ƃ��Ă�keras��tensorflow���g�p�����B  
```
python keras.py
```
�����I�v�V�����Ƃ��āA�G�|�b�N��(-e )�A���j�b�g��(-u )�A�o�b�`�T�C�Y(-b )���w��ł���B  

���}�́A�w�K���̑���(loss)�Ɨ\���덷�̕���(mae)�̗l�q�ł���B  
![sample6](docs/Figure_1_keras.png)  
  

### ����  

���}�́A�e�X�g�f�[�^�̗\���덷�̕��ςƁA�\������傫���O�ꂽ���[�X�g10�̃��X�g�ł���B
![sample7](docs/keras1.png)  

���̗�ł́A���z�u�[�X�e�B���O��A�̌��ʂ��悢���ʂɂȂ��Ă��邪�A
�j���[�����l�b�g���[�N�͏����l�̂Ƃ肩���ɂ���Ă��\���덷�͕ς��B
�j���[�����l�b�g���[�N�̕����������ʂ��o���Ƃ�������̂ŁA��T�ɂǂ��炪�悢�Ƃ͌����؂�Ȃ��B  


## ���C�Z���X  

GBR.py�͂���ɋL�ڂ���Ă��郉�C�Z���X���Q�Ƃ��Ă��������B   
  

## �Q�l�ɂ�������  

- [scikit-learn, Gradient Boosting regression example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py)
- [Deep Learning with Python�zBoston Housing Dataset��p������A���](http://liaoyuan.hatenablog.jp/entry/2018/02/03/004849)
  

## �Ɛӎ���  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.  
#### ��L��MIT���C�Z���X����̔����ł��B
