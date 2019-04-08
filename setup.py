from setuptools import setup
 
setup(
    name='MyApp',
    version='1.0',
    long_description=__doc__,
    packages=['apps','apps.v1','apps.v1.compute'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'dlib>=19.17.0',
        'Flask>=1.0.2',
        'Flask-RESTful>=0.3.7',
        'Keras>=2.2.4',
        'numpy>=1.16.2',
        'opencv-python>=4.0.0.21',
        'tensorflow>=1.13.1',
        'tensorflow-estimator>=1.13.0'
    ]
)