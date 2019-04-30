# -*- coding: utf-8 -*-
from app.extensions import api
from app.controllers import (
    health, siamese_bert_similarity
)


# Health Routes
api.add_resource(health.HealthResource, '/', '/health')


# Inference
# TODO: when done, replace SiameseBertSimilarityPlaceholder with SiameseBertSimilarity
# api.add_resource(
#     siamese_bert_similarity.SiameseBertSimilarityPlaceholder,
#     '/', '/similarities')

# TODO: remove when done
api.add_resource(
    siamese_bert_similarity.SiameseBertSimilarity,
    '/', '/similarities')
