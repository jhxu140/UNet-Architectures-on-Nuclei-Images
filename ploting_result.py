import matplotlib.pyplot as plt
import numpy as np

def ploting(datax, datay, model,model_2,num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=4, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        image_arr_nested = model_2(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()       
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization, Unet")
        ax[row_num][2].imshow(np.squeeze((image_arr_nested > 0.40)[0,:,:].astype(int)))
        ax[row_num][2].set_title("Segmented Image localization, Unet++")
        ax[row_num][3].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][3].set_title("Target image")
    plt.show()