# Memoji_Generator_with_StyleGAN2
![image](https://github.com/Megi2/Memoji_Generator_with_StyleGAN2/assets/65165556/88ba2b04-2d4c-4e78-afb8-a4c14c85aaac)
![image](https://github.com/Megi2/Memoji_Generator_with_StyleGAN2/assets/65165556/cb2ef54a-686c-4c61-a2f8-5bc122779c05)
# 📋 SUMMARY
캐릭터를 고르면 표정을 실시간 이모티콘화 해주는 IOS 프로그램인 **MEMOJI**를 사람 이미지로 부터 자동으로 생성해주는 프로그램입니다. Pytoch 기반 [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2)를 사용하여 제작하였습니다.
<br/>
[발표자료](https://drive.google.com/file/d/1FII0daszAjoj5O2mMHDplxkdilAGOSi4/view?usp=drive_link)
# REQUIRMENT
- numpy 1.23.5
- 
# ✏️ PRIOR KNOWLEDGE
## Toonify
Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains 논문의 내용을 기반으로 서술합니다.
<br/>
본 논문에서는 두 개의 StyleGAN 네트워크를 이용해 **이미지 도메인 섞기**를 제안합니다. 아래 예시에서는 각각 a) FFHQ(사람 얼굴)으로 학습된 StyleGAN에서 생성된 이미지, b) ukiyo-e(우키요에) 데이터셋으로 전이 학습한 StyleGAN으로 생성한 이미지 입니다.
<br/>
![image](https://github.com/Megi2/Memoji_Generator_with_StyleGAN2/assets/65165556/e82a3189-355f-4a3e-a051-7a3ad9b9314c)
<br/>
c~e는 두 개의 도메인을 적절하게 blending한 이미지입니다. (ex. c- 우키요에 스타일에 사람 얼굴을 블렌딩한 것)
## StyleGAN2-ADA
- 다양한 고해상도 이미지를 생성하는 GAN 모델
- 스타일 정보가 **Disentanglement** 되어 Manipulation 성능이 뛰어나다
- 이미지 x를 **latent vector** w_encoded로 인코딩하고, 이 w_encoded를
적절히 조작한 벡터를 얻고 이를 다시 최종적 이미지로 변환
<br/>

![image](https://github.com/Megi2/Memoji_Generator_with_StyleGAN2/assets/65165556/b52588ee-22b0-4de6-afe7-545194a63c29)
<br/>

![image](https://github.com/Megi2/Memoji_Generator_with_StyleGAN2/assets/65165556/ade438cc-234a-4ac1-b92a-8ec827e4bc2f)
# 😆 DATASET
다양한 피부색, 헤어스타일, 눈, 눈썹, 입 등을 가진 미모티콘 300개 이미지 데이터 제작
<br/>

![image](https://github.com/Megi2/Memoji_Generator_with_StyleGAN2/assets/65165556/45dbbc57-656f-4024-99eb-ab0937ac5100)

# Memoji_SG2_ADA_PyTorch.ipynb
## Train model
- pre-traind model인 FFHQ256(사람 얼굴 생성 모델)을 기반으로 Memoji 데이터셋을 **전이 학습**하는 과정입니다.
- dataset_path : memoji 데이터셋 경로
- resume_from : 학습을 처음 시작한다면 FFHQ256 모델의 경로, 중간에 멈췄다면 checkpoint 모델의 경로
```
dataset_path = '/content/drive/MyDrive/KHUDA_winter/Memoji.zip'
resume_from = '/content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/results2/00001-Memoji-mirror-paper256-gamma50-bg-resumeffhq256/network-snapshot-000435.pkl'
aug_strength = 0.328
train_count = 435
mirror_x = True
```
```
!python train.py --gpus=1 --cfg=$config --metrics=None --outdir=./results --data=$dataset_path --snap=$snapshot_count --resume=$resume_from --augpipe=$augs --initstrength=$aug_strength --gamma=$gamma_value --mirror=$mirror_x --mirrory=False --nkimg=$train_count
```
