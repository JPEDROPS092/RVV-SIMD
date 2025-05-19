.. RVV-SIMD documentation master file

RVV-SIMD: Biblioteca SIMD Otimizada para RISC-V Vector
======================================================

.. image:: https://img.shields.io/badge/build-passing-brightgreen
   :target: https://github.com/JPEDROPS092/sop/actions
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/bindings-pybind11-orange.svg
   :target: https://github.com/pybind/pybind11
.. image:: https://img.shields.io/badge/numpy-compatible-green.svg
   :target: https://numpy.org/
.. image:: https://img.shields.io/badge/RISC--V-RVV-red.svg
   :target: https://riscv.org/

Uma biblioteca SIMD (Single Instruction, Multiple Data) de alto desempenho otimizada para a extensão vetorial de RISC-V (RVV), com bindings Python, projetada para aplicações de machine learning e processamento de dados.

.. image:: https://riscv.org/wp-content/uploads/2020/06/riscv-color.svg
   :width: 200
   :align: center
   :alt: RISC-V Logo

Características Principais
--------------------------

* **Operações vetoriais otimizadas**: Implementações eficientes de operações básicas em vetores.
* **Operações matriciais**: Suporte a operações em matrizes, incluindo multiplicação e transposição.
* **Operações de ML**: Implementações de convolução, pooling, batch normalization e outras operações comuns em ML.
* **Bindings Python**: Interface Python completa, compatível com NumPy.
* **Benchmarks comparativos**: Ferramentas para comparar o desempenho com x86 (AVX) e ARM (NEON).
* **Implementações de fallback**: Código escalar para plataformas sem suporte a RVV, garantindo portabilidade.
* **Documentação abrangente**: Exemplos detalhados e documentação para facilitar o uso.

.. toctree::
   :maxdepth: 2
   :caption: Conteúdo:

   introducao
   arquitetura
   instalacao
   uso_cpp
   uso_python
   benchmarks
   ml_aplicacoes
   otimizacoes
   alternativas
   suporte_python
   estrutura
   contribuicoes
   faq

Índices e Tabelas
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
