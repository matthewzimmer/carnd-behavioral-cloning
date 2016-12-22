import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

class ImagePlotter:
    @staticmethod
    def plot_image(image):
        # image = mpimg.imread(X_train[0][0])
        # image = X_train[0][0]
        plt.imshow(image, interpolation='nearest')
        # plt.imshow(image)
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_images1(images, labels, cmap=None):
        gs = gridspec.GridSpec(10, 10)
        gs.update(wspace=0.01, hspace=0.02)  # set the spacing between axes.

        plt.figure(figsize=(12, 12))

        for i in range(len(images)):
            ax = plt.subplot(gs[i])

            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.set_aspect('equal')
            xlabel = "T: {0}, P: {1}".format(labels[i], None)
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

            plt.subplot(10, 10, i + 1)
            ax.imshow(images[i], cmap=cmap, interpolation='bicubic')
            # plt.axis('off')

        plt.show()

    def plot_images(images, labels, rows=5, columns=5, cls_pred=None, cmap=None):
        fig, axes = plt.subplots(rows, columns)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for i, ax in enumerate(axes.flat):
            if i >= len(images):
                break
            # Plot image.
            ax.imshow(images[i], cmap=cmap)

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(labels[i])
            else:
                xlabel = "T: {0}, P: {1}".format(labels[i], cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()