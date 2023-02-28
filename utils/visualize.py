from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

__all__ = []


def plot_matrix(Y_true,Y_predict,epoch,args):
    plt.clf()
    matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
    classes = [1,2,3,4,5,6,7,8]    # X/Y label
    confusion = confusion_matrix(Y_true,Y_predict,normalize='true')
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes,fontproperties='Times New Roman', size=8)
    plt.yticks(indices, classes,fontproperties='Times New Roman', size=8)
    plt.colorbar()
    plt.xlabel('Predicted label',fontdict={'family':'Times New Roman','size':8})
    plt.ylabel('True label',fontdict={'family':'Times New Roman','size':8})
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index,second_index,"%0.2f"%(confusion[second_index][first_index],),fontdict={'family':'Times New Roman', 'size': 8},va='center',ha='center')
#     plt.savefig(r'./draw/confusion_matrix_method.png',bbox_inches='tight', dpi=600)
    savepath = os.path.join(args.save, 'ConMatrix')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    plt.savefig(os.path.join(savepath,str(epoch)+'.jpg'),bbox_inches='tight', dpi=600)
    plt.ioff()


def plot_train(trainlosslist,trainacclist,testlosslist,testacclist,testauclist,lr_list,epoch,args):
    plt.clf()
    x1 = range(0, epoch)
    x2 = range(0, epoch)
    x3 = range(0, epoch)
    x4 = range(0, epoch)
    x5 = range(0, epoch)

    y1 = trainlosslist
    y2 = trainacclist
    y3 = testlosslist
    y4 = testacclist
    y5 = testauclist
    y6 = lr_list
    # plt.subplot(2, 1, 1)
    plt.subplot(3, 1, 1)
#     if num <= 1:
    plt.plot(x2, y2, '.-', label='train_acc', color='#FF0000')
    plt.plot(x4, y4, '.-', label='test_acc', color='#D2691E')
    plt.plot(x5, y5, '.-', label='test_auc', color='#FFC0CB')
    plt.legend(loc='upper left')
    plt.title(' METRIC vs. EPOCHS')
#     plt.xlabel('epoches')
    plt.ylabel('ACC')
    #
    plt.subplot(3, 1, 2)
#     if num <= 1:
    plt.plot(x1, y1, '.-', label='train_loss', color='#0000FF')
    plt.plot(x3, y3, '.-', label='test_loss', color='#FF0000')
    plt.legend(loc='upper right')
    plt.xlabel('EPOCHS')
    plt.ylabel('LOSS')
    # plot lr
    plt.subplot(3, 1, 3)
#     if num <= 1:
    plt.plot(x1, y6, '.-', label='lr', color='#0000FF')
    plt.legend(loc='upper right')
    plt.xlabel('EPOCHS')
    plt.ylabel('Learning rate')
    #
    fig = plt.gcf()
#     plt.show()
    savepath = os.path.join(args.save, 'plot')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    fig.savefig(os.path.join(savepath,str(epoch)+'.jpg'))
    plt.ioff()
