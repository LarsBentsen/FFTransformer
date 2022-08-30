import collections


NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = "senders"
N_NODE = "n_node"
N_EDGE = "n_edge"
GRAPH_MAPPING = "graph_mapping"
STATION_NAMES = "station_names"

GRAPH_FEATURE_FIELDS = (NODES, EDGES)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, N_NODE, N_EDGE, GRAPH_MAPPING, STATION_NAMES)


class GraphsTuple(
    collections.namedtuple("GraphsTuple",
                           GRAPH_DATA_FIELDS)):

    def __init__(self, *args, **kwargs):
        del args, kwargs
        # The fields of a `namedtuple` are filled in the `__new__` method.
        # `__init__` does not accept parameters.
        super(GraphsTuple, self).__init__()

    def replace(self, **kwargs):
        output = self._replace(**kwargs)
        return output

    def map(self, field_fn, fields=GRAPH_FEATURE_FIELDS):
        """Applies `field_fn` to the fields `fields` of the instance.
        `field_fn` is applied exactly once per field in `fields`. The result must
        satisfy the `GraphsTuple` requirement w.r.t. `None` fields, i.e. the
        `SENDERS` cannot be `None` if the `EDGES` or `RECEIVERS` are not `None`,
        etc.
        Args:
          field_fn: A callable that take a single argument.
          fields: (iterable of `str`). An iterable of the fields to apply
            `field_fn` to.
        Returns:
          A copy of the instance, with the fields in `fields` replaced by the result
          of applying `field_fn` to them.
        """
        return self.replace(**{k: field_fn(getattr(self, k)) for k in fields})
