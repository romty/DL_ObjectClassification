
import numpy as np
import os
import scipy

from matplotlib import pyplot as plt
from matplotlib import colors
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, slic, felzenszwalb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

RASTER_DATA_FILE = "data/image/Zone1_subset.tif"
TRAIN_DATA_PATH = "data/ttrain/"
TEST_DATA_PATH = "data/ttest/"



def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize all the vectors in the given directory into a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i+1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels


raster_dataset = gdal.Open(RASTER_DATA_FILE, gdal.GA_ReadOnly)
geo_transform = raster_dataset.GetGeoTransform()
proj = raster_dataset.GetProjectionRef()
n_bands = raster_dataset.RasterCount
bands_data = []
for b in range(1, n_bands+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

bands_data = np.dstack(b for b in bands_data)
print (bands_data.shape)
print ("111_Reading bands")
img = exposure.rescale_intensity(bands_data, in_range=(0, 255))
rgb_img = np.dstack([img[:, :, 3], img[:, :, 2], img[:, :, 1]])

print ("111++Do stacking ")
# plt.figure()
# plt.imshow(rgb_img)
# plt.show()


segments_quick = quickshift(rgb_img, kernel_size=7, max_dist=6, ratio=0.5)
print ("111 do segmentation with quick")
n_segments = len(np.unique(segments_quick))
print(n_segments)

cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
#plt.figure()
#plt.imshow(segments_quick, interpolation='none', cmap=cmap)
#plt.show()

band_segmentation = []
for i in range(n_bands):
    band_segmentation.append(slic(img[:, :, i], n_segments=100, compactness=10))

print ("finish slic")

const = [b.max() + 1 for b in band_segmentation]
segmentation = band_segmentation[0]
for i, s in enumerate(band_segmentation[1:]):
    segmentation += s * np.prod(const[:i+1])

_, labels = np.unique(segmentation, return_inverse=True)
segments_felz = labels.reshape(img.shape[:2])
print ("finished Felz segmentation")


cmap = colors.ListedColormap(np.random.rand(len(np.unique(segments_felz)), 3))
#plt.figure()
#plt.imshow(segments_felz, interpolation='none', cmap=cmap)


n_segments = max(len(np.unique(s)) for s in [segments_quick, segments_felz])
cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
#SHOW_IMAGES:
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
#ax1.imshow(rgb_img, interpolation='none')
#ax1.set_title('Original image')
#ax2.imshow(segments_quick, interpolation='none', cmap=cmap)
#ax2.set_title('Quickshift segmentations')
#ax3.imshow(segments_felz, interpolation='none', cmap=cmap)
#ax3.set_title('Felzenszwalb segmentations')
#plt.show()

# We choose the quick segmentation
segments = segments_felz
segment_ids = np.unique(segments)
print("Felzenszwalb segmentation. %i segments." % len(segment_ids))


rows, cols, n_bands = img.shape
files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.shp')]
classes_labels = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(TRAIN_DATA_PATH, f) for f in files if f.endswith('.shp')]
print(shapefiles)

ground_truth = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)

classes = np.unique(ground_truth)[1:]  # 0 doesn't count
len(classes)

segments_per_klass = {}
for klass in classes:
    segments_of_klass = segments[ground_truth==klass]
    segments_per_klass[klass] = set(segments_of_klass)
    print("Training segments for class %i: %i" % (klass, len(segments_per_klass[klass])))

#Disambiguation
#
# Check if there are segments which contain training pixels of different classes.
accum = set()
intersection = set()
for class_segments in segments_per_klass.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
#assert len(intersection) == 0

train_img = np.copy(segments)
threshold = train_img.max() + 1
for klass in classes:
    klass_label = threshold + klass
    for segment_id in segments_per_klass[klass]:
        train_img[train_img == segment_id] = klass_label
train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold


#plt.figure()
#cm = np.array([[ 1,  1,  1 ], [ 1,0,0], [ 1,0,1], [ 0,1,0], [ 0,1,1], [ 0,0,1]])
#cmap = colors.ListedColormap(cm)
#plt.imshow(train_img, cmap=cmap)
#plt.colorbar(ticks=[0,1,2,3,4,5])


def segment_features(segment_pixels):
    """For each band, compute: min, max, mean, variance, skewness, kurtosis"""
    features = []
    n_pixels, n_bands = segment_pixels.shape
    for b in range(n_bands):
        stats = scipy.stats.describe(segment_pixels[:,b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if n_pixels == 1:
            # scipy.stats.describe raises a Warning and sets variance to nan
            band_stats[3] = 0.0  # Replace nan with something (zero)
        features += band_stats
    return features


# This is the most heavy part of the process. It could take about half an hour to finish in a not-so-fast CPU
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    objects = []
    objects_ids = []
    for segment_label in segment_ids:
        segment_pixels = img[segments == segment_label]
        segment_model = segment_features(segment_pixels)
        objects.append(segment_model)
        # Keep a reference to the segment label
        objects_ids.append(segment_label)

    print("Created %i objects" % len(objects))


training_labels = []
training_objects = []
for klass in classes:
    class_train_objects = [v for i, v in enumerate(objects) if objects_ids[i] in segments_per_klass[klass]]
    training_labels += [klass] * len(class_train_objects)
    print("Training samples for class %i: %i" % (klass, len(class_train_objects)))
    training_objects += class_train_objects

classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)
predicted = classifier.predict(objects)

clf = np.copy(segments)

for segment_id, klass in zip(objects_ids, predicted):
    clf[clf==segment_id] = klass

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(rgb_img, interpolation='none')
ax1.set_title('Original image')
ax2.imshow(clf, interpolation='none', cmap=colors.ListedColormap(np.random.rand(len(classes_labels), 3)))
ax2.set_title('Clasification')

# Classification validation

shapefiles = [os.path.join(TEST_DATA_PATH, "%s.shp"%c) for c in classes_labels]
verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
for_verification = np.nonzero(verification_pixels)

verification_labels = verification_pixels[for_verification]
predicted_labels = clf[for_verification]

cm = metrics.confusion_matrix(verification_labels, predicted_labels)

def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    # https://gist.github.com/ClementC/acf8d5f21fd91c674808
    columnwidth = max([len(x) for x in labels])
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()

print_cm(cm, classes_labels)

print("Classification accuracy: %f" %
      metrics.accuracy_score(verification_labels, predicted_labels))

print("Classification report:\n%s" %
      metrics.classification_report(verification_labels, predicted_labels,
                                    target_names=classes_labels))