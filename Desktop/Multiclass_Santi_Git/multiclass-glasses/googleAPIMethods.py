# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:53:30 2018

@author: FraudDetectionTeam
"""
from google.cloud import storage
from google.cloud import vision
from google.protobuf import json_format
import os
import io

""" GOOGLE VISION API METHODS"""
def detect_labels(path):
    """Detects labels in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
   # print('Labels:')
    descriptions = dict()
    for label in labels:
       # print(label.description)
        descriptions[label.description.lower()]=label.score
    return descriptions
        
# [START vision_web_detection]
def detect_web(path):
    """Detects web annotations given an image."""
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_web_detection]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection
    """STOP PRINTING PLEASE
   # print(type(annotations))
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            #print('\nBest guess label: {}'.format(label.label))

    if annotations.pages_with_matching_images:
        #print('\n{} Pages with matching images found:'.format( 
        #len(annotations.pages_with_matching_images)))

        for page in annotations.pages_with_matching_images:
            #print('\n\tPage url   : {}'.format(page.url))

            if page.full_matching_images:
                #print('\t{} Full Matches found: '.format(
                       len(page.full_matching_images)))

                for image in page.full_matching_images:
                    #print('\t\tImage url  : {}'.format(image.url))

            if page.partial_matching_images:
                #print('\t{} Partial Matches found: '.format(len(page.partial_matching_images)))

                for image in page.partial_matching_images:
                    #print('\t\tImage url  : {}'.format(image.url))

    if annotations.web_entities:
        #print('\n{} Web entities found: '.format(len(annotations.web_entities)))

        for entity in annotations.web_entities:
           # print('\n\tScore      : {}'.format(entity.score))
           # print(u'\tDescription: {}'.format(entity.description))

    if annotations.visually_similar_images:
       # print('\n{} visually similar images found:\n'.format(
           # len(annotations.visually_similar_images)))

        for image in annotations.visually_similar_images:
           # print('\tImage url    : {}'.format(image.url))
       """     
    return annotations
    # [END vision_python_migration_web_detection]
# [END vision_web_detection]


def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

   # print('Number of objects found: {}'.format(len(objects)))
    objs = dict()
    for object_ in objects:
       # print('\n{} (confidence: {})'.format(object_.name, object_.score))
       # print('Normalized bounding polygon vertices: ')
        objs[object_.name.lower()] = object_.score
        #for vertex in object_.bounding_poly.normalized_vertices:
            #print(' - ({}, {})'.format(vertex.x, vertex.y))
    return objs
    """END"""
