# %% [markdown]
# ## **Three Ways of Storing and Accessing Lots of Images in Python**
# Nama : Dimas Rizky Ramadhani \
# NIM : 121450027 \
# Kelas : RC

# %%
import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("/content/cifar-10-batches-py")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")

# %% [markdown]
# Berdasarkan output, output diatas telah memuat set data pelatihan CIFAR-10, yang terdiri dari 50.000 gambar berukuran 32x32 piksel dengan 3 saluran warna (RGB). Selain itu, terdapat 50.000 label yang sesuai dengan gambar-gambar tersebut. Ini menunjukkan bahwa Anda memiliki 50.000 gambar dengan label yang sesuai untuk digunakan dalam pelatihan atau analisis lebih lanjut.

# %%
!pip install pillow

# %% [markdown]
# Menunjukkan bahwa paket Pillow telah diinstal dengan versi 9.4.0 di lingkungan Python. Ini adalah pesan yang memberi tahu bahwa tidak perlu mengunduh atau menginstal kembali paket tersebut karena sudah ada dan sudah terpasang dengan versi yang sesuai. Dengan demikian, kita dapat melanjutkan penggunaan paket Pillow dalam proyek tanpa perlu melakukan tindakan tambahan terkait instalasi.

# %% [markdown]
# ## Getting Started With LMDB

# %%
!pip install lmdb

# %% [markdown]
# lmdb, atau "Lightning Memory-Mapped Database," adalah sistem penyimpanan data yang berfokus pada kunci yang dirancang untuk mengoptimalkan kinerja dan penggunaan memori. Dengan kemampuan ini, lmdb mampu menyimpan dan mengakses data dengan kecepatan dan efisiensi yang tinggi.

# %% [markdown]
# ## Getting Started With HDF5

# %%
!pip install h5py

# %% [markdown]
# H5py adalah sebuah pustaka Python yang memberikan cara untuk berinteraksi dengan kumpulan data dalam format HDF5 (Hierarchical Data Format version 5). Format HDF5 digunakan khususnya untuk menyimpan dan mengorganisir data dalam skala besar dalam satu file tunggal, memungkinkan penggunaan yang efisien untuk pembuatan, pembacaan, dan penulisan dataset.

# %%
from pathlib import Path

disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")

# %% [markdown]
# Dalam kode tersebut, terdapat tiga variabel yang diciptakan: disk_dir, lmdb_dir, dan hdf5_dir. Masing-masing variabel didefinisikan sebagai objek Path yang mencerminkan jalur ke direktori tempat data akan disimpan. Tiga metode penyimpanan yang berbeda digunakan, yaitu penyimpanan pada disk, penyimpanan menggunakan LMDB, dan penyimpanan menggunakan HDF5.

# %%
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Dalam skrip tersebut, tujuannya adalah untuk menciptakan direktori penyimpanan data dalam tiga format yang berbeda: disk, LMDB, dan HDF5. Jika direktori-direktori tersebut belum ada, fungsi mkdir() akan digunakan untuk menciptakan mereka secara otomatis.

# %% [markdown]
# ## Storing to Disk

# %%
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])

# %% [markdown]
# Dalam fungsi store_single_disk, tugasnya adalah menyimpan gambar ke dalam format .png dan labelnya ke dalam file .csv. Data gambar, direpresentasikan sebagai array dengan dimensi (32, 32, 3), akan disimpan bersama dengan image_id dan label gambar yang sesuai.
# 
# Untuk menyimpan gambar dalam format .png, fungsi ini menggunakan modul PIL, sementara untuk menyimpan label ke dalam file .csv, digunakan modul csv.

# %% [markdown]
# ## Storing to LMDB

# %%
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)

# %% [markdown]
# Kelas CIFAR_Image digunakan untuk merepresentasikan gambar dalam dataset yang digunakan yaitu CIFAR-10. Kelas ini memiliki dua atribut utama, yaitu image yang menyimpan representasi gambar dalam bentuk byte dan label yang menyimpan label untuk gambar tersebut.

# %%
import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 1024

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

# %% [markdown]
# Fungsi store_single_lmdb bertujuan untuk menyimpan satu gambar ke dalam database LMDB (Lightning Memory-Mapped Database). Dalam fungsi ini, modul lmdb digunakan untuk membuat dan mengelola database LMDB. Setiap gambar disimpan dalam bentuk byte menggunakan modul pickle dan diindeks dengan image_id sebagai kunci.
# 
# Selain itu, Kelas CIFAR_Image digunakan untuk merepresentasikan gambar dalam dataset CIFAR-10. Kelas ini memiliki dua atribut utama: image yang menyimpan representasi gambar dalam bentuk byte, dan label yang menyimpan label untuk gambar tersebut.

# %% [markdown]
# ## Storing With HDF5

# %%
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()

# %% [markdown]
# Fungsi ini memiliki tujuan yang serupa dengan yang sebelumnya, yaitu untuk menyimpan satu gambar ke dalam format HDF5 (Hierarchical Data Format version 5). Namun, kali ini menggunakan pendekatan yang berbeda dengan menyimpan data ke dalam file HDF5.
# 
# Dalam implementasinya, fungsi ini mungkin menggunakan modul khusus untuk bekerja dengan format HDF5 dan menyimpan gambar ke dalam struktur dataset yang sesuai di dalam file HDF5.

# %% [markdown]
# ## Experiments for Storing a Single Image

# %%
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)

# %% [markdown]
# Dictionary _store_single_funcs berfungsi sebagai wadah untuk menyimpan referensi ke berbagai fungsi yang digunakan untuk menyimpan satu gambar ke dalam format yang berbeda, seperti disk, LMDB, dan HDF5. Setiap kunci dalam dictionary ini mengidentifikasi metode penyimpanan (disk, lmdb, hdf5), sedangkan nilai dari setiap kunci adalah referensi ke fungsi yang sesuai untuk metode penyimpanan tersebut. Ini memungkinkan pengguna untuk dengan mudah mengakses fungsi yang sesuai dengan metode penyimpanan yang mereka pilih tanpa perlu merinci implementasi fungsi setiap kali.

# %%
from timeit import timeit

store_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")

# %% [markdown]
# Kode di atas digunakan untuk mengukur waktu yang diperlukan untuk menyimpan satu gambar menggunakan metode penyimpanan yang sudah dibuat sebelumnya yaitu penyimpanan (disk, LMDB, dan HDF5).

# %% [markdown]
# ## Storing Many Images

# %%
def store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")

    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])

def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()

# %% [markdown]
# Fungsi store_many_disk digunakan untuk menyimpan array gambar ke dalam format file .png satu per satu, sambil juga menyimpan labelnya ke dalam file .csv. Fungsi store_many_lmdb bertugas mengelola penyimpanan array gambar dalam format LMDB, di mana setiap gambar disimpan dalam bentuk byte dengan indeks gambar sebagai kunci. Selain itu, fungsi store_many_hdf5 bertanggung jawab atas menyimpan array gambar dalam format HDF5, di mana array gambar disimpan sebagai dataset "images", sementara label disimpan sebagai dataset "meta" di dalam file HDF5.

# %% [markdown]
# ## Preparing the Dataset

# %%
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))

# %% [markdown]
# Kode di atas menggandakan jumlah gambar dalam dataset. Variabel cutoffs mengandung jumlah penggandaan yang akan dilakukan, yaitu [10, 100, 1000, 10000, 100000]. Kemudian, gambar-gambar dalam dataset images dan label-label dalam dataset akan digandakan dengan menggunakan fungsi np.concatenate((images, images), axis=0) untuk menggabungkan dataset dengan dirinya sendiri secara berulang. Ini akan menghasilkan peningkatan jumlah gambar dan label sesuai dengan jumlah penggandaan yang ditentukan dalam variabel cutoffs.

# %% [markdown]
# ## Experiment for Storing Many Images

# %%
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

from timeit import timeit

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")

# %% [markdown]
# Kode di atas bertujuan untuk mengukur waktu yang diperlukan untuk menyimpan banyak gambar, di mana jumlah gambar telah digandakan sebelumnya. Pengukuran waktu dilakukan menggunakan 3 metode penyimpanan yang berbeda, yaitu disk, LMDB, dan HDF5. Dengan menggunakan fungsi-fungsi yang telah ditentukan sebelumnya untuk setiap metode penyimpanan, waktu yang diperlukan untuk menyimpan gambar-gambar tersebut akan diukur dan dicatat. Ini memberikan pemahaman tentang kinerja relatif dari masing-masing metode penyimpanan dalam konteks jumlah gambar yang besar.

# %%
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)

# %% [markdown]
# Fungsi diatas digunakan untuk membuat dua plot yang menunjukkan waktu penyimpanan untuk setiap metode penyimpanan (disk, LMDB, HDF5) terhadap berbagai ukuran dataset.
# 
# Plot pertama menunjukkan waktu penyimpanan dalam skala linier yang menunjukkan grafik durasi penyimpanan gambar dengan format PNG, LMDB, dan HDF5. Grafik menunjukkan bahwa durasi penyimpanan gambar meningkat seiring dengan jumlah gambar yang disimpan. Format PNG memiliki durasi penyimpanan gambar yang paling lama, diikuti oleh format LMDB dan HDF5.
# 
# Sementara plot kedua menunjukkan waktu penyimpanan dalam skala logaritmik yang menunjukkan grafik durasi penyimpanan gambar dengan format PNG, LMDB, dan HDF5. Grafik menunjukkan bahwa durasi penyimpanan gambar meningkat seiring dengan jumlah gambar yang disimpan. Format PNG memiliki durasi penyimpanan gambar yang paling lama, diikuti oleh format LMDB dan HDF5.

# %%
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)

# %% [markdown]
# Grafik tersebut juga mengkonfirmasi bahwa format PNG memiliki durasi penyimpanan gambar yang terpanjang, diikuti oleh format LMDB dan HDF5. Format PNG tidak melakukan kompresi data saat menyimpan gambar, menghasilkan ukuran file yang lebih besar dan durasi penyimpanan yang lebih lama. Di sisi lain, format LMDB yang dioptimalkan untuk penyimpanan gambar menghasilkan ukuran file yang lebih kecil dan durasi penyimpanan yang lebih cepat dibandingkan dengan format PNG. Sedangkan, format HDF5, meskipun dirancang untuk data ilmiah, memiliki ukuran file yang lebih besar dan durasi penyimpanan yang lebih lama dibandingkan dengan format LMDB karena tidak dioptimalkan khusus untuk gambar.

# %% [markdown]
# ## Reading a Single Image

# %%
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))

    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])

    return image, label

# %% [markdown]
# Fungsi read_single_disk memiliki tujuan untuk membaca sebuah gambar dan label yang telah disimpan di disk. Fungsi ini berguna untuk membaca kembali gambar dan label yang telah disimpan sebelumnya ke dalam disk, sehingga memungkinkan untuk melakukan operasi baca ulang terhadap data yang telah disimpan sebelumnya.

# %% [markdown]
# ## Reading From LMDB

# %%
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()

    return image, label

# %% [markdown]
# Fungsi read_single_lmdb bertugas membaca sebuah gambar beserta labelnya yang tersimpan di dalam LMDB. Fungsi ini mengembalikan array gambar dan label, sehingga memungkinkan untuk membaca kembali gambar dan label yang telah disimpan di dalam LMDB untuk penggunaan selanjutnya.

# %% [markdown]
# ## Reading From HDF5

# %%
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label

# %% [markdown]
# Fungsi read_single_hdf5 berfungsi untuk membaca sebuah gambar beserta labelnya yang tersimpan di dalam file HDF5. Fungsi ini digunakan untuk membaca kembali gambar dan label yang telah disimpan di dalam file HDF5, memungkinkan penggunaan kembali data tersebut untuk pengolahan atau analisis lebih lanjut.

# %%
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)

# %% [markdown]
# _read_single_funcs adalah sebuah kamus (dictionary) yang mengandung fungsi-fungsi untuk membaca sebuah gambar beserta labelnya dari berbagai jenis penyimpanan yang telah dibuat sebelumnya. Setiap kunci dalam kamus ini mengidentifikasi jenis penyimpanan, sementara nilainya adalah fungsi yang sesuai untuk membaca gambar dan label dari jenis penyimpanan tersebut. Ini memungkinkan untuk dengan mudah memanggil fungsi yang sesuai tergantung pada jenis penyimpanan yang digunakan, tanpa perlu merinci implementasi fungsi setiap kali.

# %% [markdown]
# ## Experiment for Reading a Single Image

# %%
from timeit import timeit

read_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")

# %% [markdown]
# Kode tersebut bertujuan untuk mengukur waktu pembacaan (read) sebuah gambar beserta labelnya dari penyimpanan disk, LMDB, dan HDF5. Tujuannya adalah untuk membandingkan performa pembacaan dari berbagai jenis penyimpanan tersebut. Dengan melakukan pengukuran waktu, kita dapat memahami seberapa cepat atau lambat proses pembacaan data dari masing-masing jenis penyimpanan, dan dengan demikian, dapat memilih jenis penyimpanan yang paling sesuai untuk kebutuhan spesifik dalam suatu aplikasi atau tugas.

# %% [markdown]
# # Reading Many Images

# %% [markdown]
# ## Adjusting the Code for Many Images

# %%
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))

    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)

# %% [markdown]
# Fungsi-fungsi read_many_disk, read_many_lmdb, dan read_many_hdf5 digunakan untuk membaca banyak gambar beserta labelnya dari berbagai jenis penyimpanan yang telah dibuat sebelumnya. Sementara itu, _read_many_funcs digunakan sebagai wadah untuk menyimpan fungsi-fungsi pembacaan yang sesuai untuk setiap jenis penyimpanan, seperti disk, LMDB, dan HDF5. Dengan demikian, memungkinkan untuk dengan mudah memanggil fungsi pembacaan yang sesuai tergantung pada jenis penyimpanan yang digunakan, tanpa perlu merinci implementasi fungsi setiap kali.

# %% [markdown]
# ## Experiment for Reading Many Images

# %%
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")

# %% [markdown]
# Dari hasil yang tercetak, dapat disimpulkan bahwa waktu yang diperlukan untuk membaca dari disk (disk) cenderung meningkat seiring dengan peningkatan jumlah gambar. Selain itu, waktu yang diperlukan untuk membaca dari LMDB (lmdb) dan HDF5 (hdf5) juga meningkat, namun LMDB cenderung lebih cepat daripada HDF5, terutama saat jumlah gambar semakin besar. Secara keseluruhan, penggunaan LMDB dan HDF5 memberikan waktu baca yang lebih cepat dibandingkan dengan membaca langsung dari disk, terutama saat jumlah gambar cukup besar.

# %%
disk_x_r = read_many_timings["disk"]
lmdb_x_r = read_many_timings["lmdb"]
hdf5_x_r = read_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Read time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Log read time",
    log=True,
)

# %% [markdown]
# Kode tersebut menghasilkan dua plot yang membandingkan waktu yang diperlukan untuk membaca sejumlah gambar. Dari Gambar 1 dan Gambar 2, terlihat bahwa format PNG memiliki durasi penyimpanan gambar yang paling lama, diikuti oleh format LMDB dan HDF5. Selain itu, terlihat juga bahwa durasi penyimpanan gambar meningkat seiring dengan jumlah gambar yang disimpan, sesuai dengan trend yang dijelaskan sebelumnya.

# %%
plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r, disk_x, lmdb_x, hdf5_x],
    [
        "Read PNG",
        "Read LMDB",
        "Read HDF5",
        "Write PNG",
        "Write LMDB",
        "Write HDF5",
    ],
    "Number of images",
    "Seconds",
    "Log Store and Read Times",
    log=False,
)

# %% [markdown]
# Grafik di atas memperlihatkan durasi penyimpanan dan durasi pembacaan data. Dapat diamati bahwa format PNG membutuhkan waktu yang lebih lama dibandingkan dengan format lainnya.


