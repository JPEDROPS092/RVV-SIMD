Instalação
==========

Esta seção descreve como instalar a biblioteca RVV-SIMD em diferentes ambientes.

Pré-requisitos
-------------

Para compilar e usar a biblioteca RVV-SIMD, você precisará dos seguintes componentes:

* **Toolchain RISC-V**: Compilador (GCC ou Clang) com suporte à extensão vetorial (`-march=rv64gcv` ou similar).
* **CMake**: Versão 3.10 ou superior.
* **Python**: Versão 3.6 ou superior (necessário apenas para os bindings Python).
* **pybind11**: Biblioteca C++ para criar bindings Python (geralmente incluída como submódulo ou baixada pelo CMake).
* **NumPy**: Biblioteca Python para manipulação de arrays (necessária para os exemplos e testes Python).

Compilando a partir do Código Fonte
----------------------------------

1. **Clone o repositório:**

   .. code-block:: bash

       git clone https://github.com/JPEDROPS092/sop.git
       cd sop
       # Opcional: Inicializar submódulos (se pybind11 for um submódulo)
       # git submodule update --init --recursive

2. **Crie um diretório de build e configure com CMake:**

   .. code-block:: bash

       mkdir build && cd build
       
       # Para build padrão (detectará RVV se o toolchain suportar):
       cmake ..
       
       # Para forçar build com RVV (requer toolchain compatível):
       # cmake .. -DRVV_SIMD_FORCE_RVV=ON
       
       # Para forçar build com fallback (útil para testes em x86/ARM):
       # cmake .. -DRVV_SIMD_FORCE_FALLBACK=ON
       
       # Para habilitar build dos bindings Python:
       # cmake .. -DRVV_SIMD_BUILD_PYTHON=ON

3. **Compile a biblioteca:**

   .. code-block:: bash

       make -j$(nproc) # Compila em paralelo

4. **(Opcional) Instale a biblioteca:**

   .. code-block:: bash

       sudo make install # Instala headers e biblioteca no sistema

Instalando os Bindings Python
----------------------------

Se você habilitou a opção `DRVV_SIMD_BUILD_PYTHON=ON` no CMake e compilou:

1. **Navegue até o diretório Python:**

   .. code-block:: bash

       cd ../python # A partir do diretório 'build'

2. **Instale o pacote Python em modo editável:**

   .. code-block:: bash

       pip install -e .

   Isso cria um link para o módulo compilado no diretório `build`, permitindo que você importe `rvv_simd` em Python.

Instalação via pip (quando disponível)
-------------------------------------

Para uma instalação mais simples, você pode usar pip (quando disponível):

.. code-block:: bash

    pip install rvv-simd

Observe que esta instalação via pip usará a implementação de fallback em sistemas não-RISC-V. Para obter o máximo desempenho em hardware RISC-V com extensão vetorial, é recomendável compilar a partir do código fonte.

Verificando a Instalação
----------------------

Para verificar se a biblioteca foi instalada corretamente, você pode executar os testes incluídos:

.. code-block:: bash

    # A partir do diretório build
    make test
    
    # Para testes Python
    cd ../python
    pytest

Você também pode verificar se a biblioteca está funcionando corretamente executando um dos exemplos incluídos:

.. code-block:: bash

    # Exemplos C++
    cd ../examples
    ./vector_example
    
    # Exemplos Python
    cd ../python/examples
    python vector_operations.py

Configurações Avançadas
---------------------

A biblioteca RVV-SIMD oferece várias opções de configuração que podem ser ajustadas durante a compilação:

* **RVV_SIMD_FORCE_RVV**: Força o uso da implementação RVV, mesmo em plataformas não-RISC-V (útil para cross-compilação).
* **RVV_SIMD_FORCE_FALLBACK**: Força o uso da implementação de fallback, mesmo em plataformas RISC-V com suporte a RVV (útil para testes).
* **RVV_SIMD_BUILD_PYTHON**: Habilita a compilação dos bindings Python.
* **RVV_SIMD_BUILD_TESTS**: Habilita a compilação dos testes.
* **RVV_SIMD_BUILD_BENCHMARKS**: Habilita a compilação dos benchmarks.
* **RVV_SIMD_BUILD_EXAMPLES**: Habilita a compilação dos exemplos.

Exemplo de uso:

.. code-block:: bash

    cmake .. -DRVV_SIMD_BUILD_PYTHON=ON -DRVV_SIMD_BUILD_TESTS=ON -DRVV_SIMD_BUILD_BENCHMARKS=ON

Solução de Problemas
------------------

Problemas Comuns
^^^^^^^^^^^^^^^

1. **Erro de compilação relacionado a RVV**:
   
   Se você encontrar erros relacionados a intrínsecos RVV, verifique se seu compilador suporta a extensão vetorial RISC-V e se você está usando as flags de compilação corretas.

   .. code-block:: bash

       # Verifique a versão do GCC
       riscv64-unknown-linux-gnu-gcc --version
       
       # Verifique se o compilador suporta a extensão vetorial
       echo "int main() { return 0; }" | riscv64-unknown-linux-gnu-gcc -march=rv64gcv -x c -c -o /dev/null - && echo "RVV suportado" || echo "RVV não suportado"

2. **Erro ao importar o módulo Python**:
   
   Se você encontrar erros ao importar o módulo Python, verifique se o módulo foi compilado corretamente e se está no PYTHONPATH.

   .. code-block:: bash

       # Verifique se o módulo está instalado
       pip list | grep rvv-simd
       
       # Verifique se o módulo pode ser importado
       python -c "import rvv_simd; print(rvv_simd.get_version())"

3. **Desempenho abaixo do esperado**:
   
   Se o desempenho estiver abaixo do esperado, verifique se a biblioteca está realmente usando a implementação RVV e não a implementação de fallback.

   .. code-block:: bash

       # Em C++
       if (rvv_simd::is_rvv_supported()) {
           std::cout << "RVV está sendo usado" << std::endl;
       } else {
           std::cout << "Implementação de fallback está sendo usada" << std::endl;
       }
       
       # Em Python
       import rvv_simd as rv
       print(f"RVV suportado: {'Sim' if rv.is_rvv_supported() else 'Não'}")

Obtendo Ajuda
^^^^^^^^^^^

Se você encontrar problemas que não consegue resolver, você pode:

* Abrir uma issue no GitHub: https://github.com/JPEDROPS092/sop/issues
* Consultar a documentação online: https://rvv-simd.readthedocs.io/
* Entrar em contato com a equipe de desenvolvimento: contato@rvv-simd.org