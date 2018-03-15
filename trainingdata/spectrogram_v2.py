import librosa as lr
import scipy.misc as misc
import os

mp3_path = 'trainingdata/'
image_path = 'spectrogramsV2/'

def mp3_to_img(path, height=192, width=192):
    signal, sr = lr.load(path, res_type='kaiser_fast')
    hl = signal.shape[0]//(width*1.1) #this will cut away 5% from start and end
    spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
    img = lr.logamplitude(spec)**2
    start = (img.shape[1] - width) // 2
    return img[:, start:start+width]

imgpath = mp3_path+"/000kouqjfnk.mp3"
img = mp3_to_img(imgpath)

misc.imsave('test.jpg',img)

file = open('trainingData.csv', 'r')

processed = os.listdir(image_path)
processed = [x[:-4] for x in processed if '.jpg' in x]

for iter, line in enumerate(
        file.readlines()[1:]):  # first line of traininData.csv is header (only for trainingData.csv)
    filepath = line.split(',')[0]
    filename = filepath[:-4]

    if filename not in processed:
        path = image_path+'/'+filename+'.jpg'
        img = mp3_to_img(mp3_path+'/'+filepath)
        misc.imsave(path, img)
