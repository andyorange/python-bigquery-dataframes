"""Utilities for meta-estimators"""
# Author: Joel Nothman
#         Andreas Mueller
# License: BSD


from abc import ABCMeta

from bigframes import constants
from third_party.bigframes_vendored.sklearn.base import BaseEstimator


class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for classifiers composed of named estimators."""


class ColumnTransformer(_BaseComposition):
    """Applies transformers to columns of BigQuery DataFrames.

    This estimator allows different columns or column subsets of the input
    to be transformed separately and the features generated by each transformer
    will be concatenated to form a single feature space.
    This is useful for heterogeneous or columnar data, to combine several
    feature extraction mechanisms or transformations into a single transformer.

    Args:
        transformers:
            List of (name, transformer, columns) tuples specifying the transformer
            objects to be applied to subsets of the data.
    """

    def fit(
        self,
        X,
    ):
        """Fit all transformers using X.

        Args:
            X (bigframes.dataframe.DataFrame or bigframes.series.Series):
                The Series or DataFrame of shape (n_samples, n_features). Training vector,
                where `n_samples` is the number of samples and `n_features` is
                the number of features.

        Returns:
            ColumnTransformer: Fitted estimator.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def transform(
        self,
        X,
    ):
        """Transform X separately by each transformer, concatenate results.

        Args:
            X (bigframes.dataframe.DataFrame or bigframes.series.Series):
                The Series or DataFrame to be transformed by subset.

        Returns:
            bigframes.dataframe.DataFrame: Transformed result.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
