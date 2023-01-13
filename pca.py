import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class pca_faces:
    def __init__(self):
        df = pd.read_csv('face_data.csv')
        self.label = df['target']
        self.persons = df['target'].nunique()
        self.data = df.drop(labels='target',axis=1)
        self.data = np.array(self.data)
        self.sh = self.data.shape
        self.svds = []
        self.offset = np.array([])
        self.pr_comp = self.pc()

    def plot_img(self):
        faces = 4
        np.random.seed(5)
        img = np.random.choice(self.data.shape[0], size=faces**2, replace=False)
        img = np.sort(img)

        fig, axes = plt.subplots(faces,faces)
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.data[int(i/faces)*10+(i%faces),:].reshape(64,64), cmap='gray')
            #self.data[img[i],:].reshape(64,64), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def plot_img_pc(self,person=0):
        faces = 4
        cmp = self.pr_comp
        VT = self.svds[2]
        prec = [1, 20, 100, 400]
        fig, axes = plt.subplots(faces, faces)
        for i, ax in enumerate(axes.flat):
            #face_data = (cmp[int(i / faces) * 10 + (i % faces), :] @ VT) + self.offset
            pr = prec[int(i/faces)]
            face_data = (cmp[person*10+(i%faces),:pr] @ VT[:pr,:]) + self.offset
            ax.imshow(face_data.reshape(64, 64), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def pc_eig(self):
        mean = np.average(self.data, axis=1)
        centered = self.data - (np.ones((self.sh[0],1)) @ np.reshape(mean,(1,self.sh[1])))
        cov = (1/self.sh[0]) * np.transpose(centered) @ centered
        eigvalues, eigvectors = np.linalg.eig(cov)
        order = np.argsort(eigvalues)
        eigvalues = np.diag(eigvalues[order])
        eigvectors = eigvectors[:,order]
        pc = centered @ eigvectors
        return pc

    def pc_svd(self):
        mean = np.average(self.data, axis=0)
        self.offset = mean
        centered = self.data - (np.ones((self.sh[0], 1)) @ np.reshape(mean, (1, self.sh[1])))
        U, sigma, VT = np.linalg.svd(centered, full_matrices=False)
        self.svds = [U, sigma, VT]
        pc = centered @ np.transpose(VT) #U @ np.diag(sigma)
        return pc

    def pc(self):
        #self.pr_comp = self.pc_svd()
        return self.pc_svd()

    def avr_faces(self, cmp=None, VT=None):
        if cmp.any() == None or VT.any() == None:
            cmp = self.pr_comp
            VT = self.svds[2]
        avr_face = np.average(cmp,axis=0) @ VT
        plt.imshow(avr_face.reshape(64,64),cmap='gray')
        plt.show()

    def reduce_dim(self, ratio=0.95):
        U, sigma, VT = self.svds
        eigv = sigma**2
        var_ratio = eigv / np.sum(eigv)
        acc_var_ratio = np.cumsum(var_ratio)
        n_cmp = 1 + np.where(acc_var_ratio>=ratio)[0][0]
        red_cmp = U[:,:n_cmp] @ np.diag(sigma[:n_cmp])
        return red_cmp, VT[0:n_cmp,:]

    def n_first_pcs(self, ns=None):
        if ns == None:
            ns = [len(self.svds[1])]
        U, sigma, VT = self.svds
        width = int(np.sqrt(len(ns)))
        fig, ax = plt.subplots(width,int(len(ns)/width))
        axes = ax.flat
        fig.suptitle('Average of faces using PCs', fontsize='large')
        vars = np.zeros([len(ns),2])
        for i, n in enumerate(ns):
            red_cmp = U[:,:n] @ np.diag(sigma[:n])
            avr_face = np.average(red_cmp, axis=0) @ VT[:n,:] +self.offset #HERE
            axes[i].imshow(avr_face.reshape(64, 64), cmap='gray')
            axes[i].set_title('First ' + str(n) + ' PCs')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            vars[i,:]=np.array([n,self.var_n_pcs(n)])
        plt.show()
        return ns

    def pcs_with_var(self, vars=[1.0]):
        U, sigma, VT = self.svds
        # Compute variances
        eigv = sigma**2
        var_ratio = eigv / np.sum(eigv)
        acc_var_ratio = np.cumsum(var_ratio)
        # Prepare plotting
        width = int(np.sqrt(len(vars)))
        fig, ax = plt.subplots(width,int(len(vars)/width))
        axes = ax.flat
        fig.suptitle('Average of faces using PCs', fontsize='large')
        for i, var in enumerate(vars):
            var = round(var,ndigits=2)
            n = 1 + np.where(acc_var_ratio>=var)[0][0]
            # Discard unnecessary PCs
            red_cmp = U[:,:n] @ np.diag(sigma[:n])
            # Averaging
            avr_face = np.average(red_cmp, axis=0) @ VT[0:n,:]
            # Plot
            axes[i].imshow(avr_face.reshape(64, 64), cmap='gray')
            axes[i].set_title(str(int(var*100))+' % of total variance')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        plt.show()
        return n

    def var_n_pcs(self,n):
        lam = self.svds[1] ** 2
        lam = (1 / np.sum(lam)) * lam
        acc_var = np.cumsum(lam)
        return acc_var[n-1]

    def plot_nth_pc(self,ns=[0]):
        U, sigma, VT = self.svds
        width = int(np.sqrt(len(ns)))
        fig, ax = plt.subplots(width,int(len(ns)/width))
        axes = ax.flat
        fig.suptitle('Principal components', fontsize='large')
        for i, n in enumerate(ns):
            title = str(n+1)+'th PC'
            if n==0:
                title = '1st PC'
            elif n==1:
                title = '2nd PC'
            elif n==2:
                title = '3rd PC'

            red_cmp = U[:,n] * sigma[n]
            avr_face = np.average(red_cmp, axis=0) * VT[n, :]
            axes[i].imshow(avr_face.reshape(64, 64), cmap='gray')
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        plt.show()
        return ns


    def plot_var(self):
        # Calculate variance of each PC
        lam = self.svds[1]**2
        lam = (1/np.sum(lam))*lam
        # Calculate accumulated variance of the first n PCs
        acc_var = np.cumsum(lam)
        # Plot 0
        fig, axes = plt.subplots(1,2)
        axes[0].plot(lam)
        axes[0].set_title('Ratio of variance for each PC')
        axes[0].margins(0)
        axes[0].set_box_aspect(1)
        axes[0].set_xlabel('# PC')
        axes[0].set_ylabel('ratio of variance')
        # Plot 1
        axes[1].set_title('Accumulated ratio of variance')
        axes[1].margins(0)
        axes[1].set_box_aspect(1)
        axes[1].set_xlabel('# PC')
        axes[1].set_ylabel('ratio of variance')
        # Plot certain levels of variance
        for ratio in [0.6,0.9]:
            cmp = np.where(acc_var >= ratio)[0][0]
            axes[1].plot([cmp,cmp],[0,1],'k', linewidth=1)
            axes[1].plot([0,acc_var.size],[acc_var[cmp],acc_var[cmp]],'k', linewidth=1)
            # Add ticks with labels on axes
            axes[1].set_xticks(list(axes[1].get_xticks()) + [cmp])
            axes[1].set_yticks(list(axes[1].get_yticks()) + [acc_var[cmp]])
        axes[1].plot(acc_var)
        plt.show()

    def show_greyscale(self):
        gr = np.linspace(-1.0,1.0,201)
        gr = gr.reshape(gr.size,1)
        plt.imshow(gr, cmap='gray')
        plt.show()


pf = pca_faces()
#pf.plot_img()
#pf.plot_img_pc()
#pf.show_greyscale()
pf.n_first_pcs(list(range(1,5))+list(range(10,171,20))+list(range(200,401,100)))
#pf.pcs_with_var(np.linspace(0.1,1,9))
#pf.plot_nth_pc(ns=list(range(9)))
