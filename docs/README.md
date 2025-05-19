# Documentação RVV-SIMD

Esta é a documentação oficial da biblioteca RVV-SIMD, uma biblioteca SIMD otimizada para a extensão vetorial RISC-V (RVV).

## Construindo a Documentação

Para construir a documentação localmente, siga os passos abaixo:

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Construa a documentação:

```bash
sphinx-build -b html . _build/html
```

3. Visualize a documentação:

```bash
python -m http.server 8000 --directory _build/html
```

Acesse http://localhost:8000 no seu navegador para visualizar a documentação.

## Hospedando no Read the Docs

Esta documentação está configurada para ser hospedada no Read the Docs. Para configurar o seu próprio projeto:

1. Crie uma conta no [Read the Docs](https://readthedocs.org/)
2. Importe o seu repositório
3. Configure o projeto para usar o arquivo `.readthedocs.yml` na raiz do repositório

## Estrutura da Documentação

- `index.rst`: Página principal da documentação
- `introducao.rst`: Introdução à biblioteca
- `arquitetura.rst`: Arquitetura da biblioteca
- `instalacao.rst`: Instruções de instalação
- `uso_cpp.rst`: Uso da biblioteca em C++
- `uso_python.rst`: Uso da biblioteca em Python
- `faq.rst`: Perguntas frequentes
- `conf.py`: Configuração do Sphinx
- `requirements.txt`: Dependências para construir a documentação
- `.readthedocs.yml`: Configuração do Read the Docs

## Contribuindo com a Documentação

Contribuições para a documentação são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para suas alterações
3. Faça as alterações na documentação
4. Envie um pull request

## Licença

Esta documentação está licenciada sob a licença MIT, assim como o código da biblioteca.