import json


class JsonlReader:

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, encoding="utf-8", errors="ignore") as rf:
            for jsonl in rf:
                jsonl = jsonl.strip()
                if not jsonl:
                    continue
                yield json.loads(jsonl)


class TriplesReader:

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, encoding="utf-8", errors="ignore") as rf:
            for tsvl in rf:
                tsvl = tsvl.strip()
                if not tsvl:
                    continue
                yield tsvl.split("\t")


def trip_dict(triples_file):
    trip2idx = dict()
    idx = 0
    with open(triples_file) as rf:
        for tsvl in rf:
            tsvl = tsvl.strip()
            if not tsvl:
                continue
            trip2idx[tsvl.lower()] = idx
            idx += 1
    return trip2idx


def trip_set(triples_file):
    t_set = set()
    with open(triples_file, 'r') as filehandle:
        raw_trips = [current_place.rstrip() for current_place in filehandle.readlines()]
        for trip in raw_trips:
            if len(trip) < 2: continue
            trip = trip[3:-2]
            h, r, t = trip.split('\\t')
            hrt = '_'.join([h, r, t])
            t_set.add(hrt.lower())
        return t_set


def read_relations(relations_file):
    print("Reading relations file: ", relations_file)
    relation2idx = dict()
    idx = 0
    with open(relations_file) as rf:
        for relation in rf:
            relation = relation.strip().lower()
            if not relation:
                continue
            relation2idx[relation] = idx
            idx += 1
    return relation2idx


def idx_to_rel(relations_file):
    print("Reading relations file: ", relations_file)
    idx2rel = dict()
    idx = 0
    with open(relations_file) as rf:
        for relation in rf:
            relation = relation.strip().lower()
            if not relation:
                continue
            idx2rel[str(idx)] = relation
            idx += 1
    return idx2rel


def read_entities(entities_file):
    entity2idx = dict()
    idx = 0
    with open(entities_file) as rf:
        for entity in rf:
            entity = entity.strip()
            if not entity:
                continue
            entity2idx[entity] = idx
            idx += 1
    return entity2idx


def idx_to_ent(entities_file):
    idx_to_ent = dict()
    idx = 0
    with open(entities_file) as rf:
        for entity in rf:
            entity = entity.strip()
            if not entity:
                continue
            idx_to_ent[str(idx)] = entity
            idx += 1
    return idx_to_ent

mongodump --uri "mongodb://admin:admin-sandbox@ucsd-sandbox.w8zcb.mongodb.net/mba-sandbox" -o ~/Sandbox/mba-backups
#