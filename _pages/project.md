---
layout: archive
author_profile: true
title: "List of My Projects"
permalink: /project/
---  



{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
{% if tag == "project" %}
  {% assign posts = group_items[forloop.index0] %}

  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
  {% endif %}
{% endfor %}


              