# micelle_scattering
Models the normalized scattering intensity profiles of various shapes of a micelle at low concentrations.

This package contains 6 classes of a micelle: sphere, spheroid, cylinder, worm, disk, and vesicle. These classes take the following 5 parameters as their inputs: the radius, the elongation coefficient, the radius of gyration of the corona, the aggregation number, the ratio betwen the corona and core scattering lengths.

In essence, this package simulates the scattering profiles of different types of micelles at low concentration (where S(q) is approximately 1). Using the given 5 parameters, the package returns a normalized scattering intensity data for a given set of q values.

This package was developed with the aim to generate simulated data for machine learning applications.
