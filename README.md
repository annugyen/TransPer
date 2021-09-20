# Description
An explainability framework for highlighting decision-relevant input data using layerwise relevance propagation.

# TransPer

Transper is a Python framework for explaining recommender decisions based on neural network decisions via the the Layerwise Relevance Propagation (LRP).

## Setup
* create virtual environment: `$ python3 -m venv venv`
* activate virtual environment: `$ source venv/bin/activate`
* install requirements: `$ pip install -r requirements.txt`

## Usage

For privacy reasons, the user data from the associated paper cannot be used. However, in Example.py we present a fictitious use case (see SampleModel.png) that explains the idea behind TransPer and facilitates an installation in a real-world scenario.

* **[Anna Nguyen](https://www.aifb.kit.edu/web/Anna_Nguyen/en)**
* **[Franz Krause](https://www.aifb.kit.edu/web/Franz_Krause/en)**
* **[Daniel Hagenmayer](https://www.aifb.kit.edu/web/Daniel_Hagenmayer/en)**
* **[Michael Färber](https://www.aifb.kit.edu/web/Michael_Färber/en)**

## Paper

[Nguyen, Anna; Weller, Tobias; Färber, Michael; Sure-Vetter, York. "Quantifying Explanations of Neural Networks in E-Commerce Based on LRP."](https://doi.org/10.1007/978-3-030-86517-7\_16)
```
@inproceedings{DBLP:conf/pkdd/NguyenKHF21,
  author    = {Anna Nguyen and
               Franz Krause and
               Daniel Hagenmayer and
               Michael F{\"{a}}rber},
  editor    = {Yuxiao Dong and
               Nicolas Kourtellis and
               Barbara Hammer and
               Jos{\'{e}} Antonio Lozano},
  title     = {Quantifying Explanations of Neural Networks in E-Commerce Based on
               {LRP}},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Applied Data
               Science Track - European Conference, {ECML} {PKDD} 2021, Bilbao, Spain,
               September 13-17, 2021, Proceedings, Part {V}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12979},
  pages     = {251--267},
  publisher = {Springer},
  year      = {2021},
  url       = {https://doi.org/10.1007/978-3-030-86517-7\_16},
  doi       = {10.1007/978-3-030-86517-7\_16}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
