# Quick Start Guide - Librarian

## Setup (5 minutes)

1. **Install dependencies:**
   ```powershell
   python setup.py
   ```

2. **Add your OpenAI API key:**
   - Open `.env` file
   - Replace `your_openai_api_key_here` with your actual API key

3. **Build the knowledge base:**
   ```powershell
   python main.py build
   ```
   This will process all PDFs in the Projects folder (takes a few minutes).

4. **Start searching:**
   ```powershell
   python cli.py
   ```

## Quick Commands

### In CLI mode:
```
search retaining wall foundation
list
stats
help
```

### From command line:
```powershell
python main.py search "bridge seismic design"
python main.py list
python main.py stats
```

## Example Searches

- `search foundation design seismic`
- `search retaining wall soil pressure`
- `search bridge deck reinforcement`
- `search tunnel boring groundwater`
- `search station platform structural`

## Need Help?

- Type `help` in CLI mode
- See [README.md](README.md) for detailed documentation
- Check `.env` file for correct API key

## Common Issues

**"No OpenAI API key found"**
- Edit `.env` file and add your API key

**"No results found"**
- Run `build` command first to index PDFs

**"No PDF files found"**
- Add PDF files to the `Projects` folder
