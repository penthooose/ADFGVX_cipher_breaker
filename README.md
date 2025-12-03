# ADFGVX Cipher Breaker

A Python implementation for encrypting, decrypting, and breaking the ADFGVX cipher using hill-climbing and n-gram scoring techniques.

Related paper: [Breaking the ADFGVX Cipher — From Theory to Practical Cryptanalysis](docs/Breaking_the_ADFGVX_Cipher-From_Theory_to_Practical_Cryptanalysis.pdf) — the accompanying PDF is included in the `docs/` folder and explains the general approach, history, and algorithms of this project.

---

## Quick Start

### Encrypt / Decrypt Messages

```python
from cipher_implementation import create_polybius_square, encrypt_message, decrypt_message

# Create Polybius square from alphabet (36 chars: A-Z + 0-9)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
polybius_map, inverse_map = create_polybius_square(alphabet)

# Encrypt
ciphertext = encrypt_message("HELLO WORLD", polybius_map, column_key="MATH")

# Decrypt
plaintext = decrypt_message(ciphertext, inverse_map, column_key="MATH")
```

### Run Example

```python
from cipher_implementation import run_example
run_example()  # Uses default plaintext, key, and alphabet
run_example(plaintext="SECRET MESSAGE", key="CRYPTO", alphabet="QWERTYUIOPASDFGHJKLZXCVBNM0123456789")
```

---

## Breaking Ciphers

### From `messages.json` (Recommended)

Store your ciphertexts in `auxiliary/messages.json` and break by entry ID:

```python
from cipher_breaker import break_cipher_from_file
break_cipher_from_file(entry_id=1)  # Breaks entry "1" from messages.json
```

Requires that respective ciphertexts were stored at this entry. They can be created with custom plaintexts, alphabets, and keys, using `fill_msg_dict()` in `cipher_implementation`.

### Direct Breaking

```python
from cipher_breaker import break_cipher

break_cipher(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    column_key="UNKNOWNKEY",  # Optional hint
    plaintexts=["KNOWN PLAINTEXT"],  # For generating test ciphertexts
    key_lengths=[10, 11, 12]  # Candidate lengths to try
)
```

---

## Utility Functions

### Generate Ciphertexts for `messages.json`

```python
from cipher_implementation import fill_msg_dict
fill_msg_dict(entry_id=1)  # Encrypts plaintexts and stores ciphertexts
```

### Get All Possible Key Orders

```python
from cipher_implementation import get_all_possible_key_orders
get_all_possible_key_orders("MATH", print_output=True)
```

---

## Configuration

Some important Key settings in `cipher_breaker.py` → `CONFIG` dict:

| Setting                       | Description                         |
| ----------------------------- | ----------------------------------- |
| `key_search_only`             | Skip alphabet optimization (faster) |
| `language`                    | `"EN"` or `"DE"` for scoring        |
| `restarts` / `max_iterations` | Search depth                        |
| `key_search_workers`          | Parallel workers                    |

---

## Documentation

- **[Cipher Breaker Doc](docs/cipher_breaker_doc.md)** — Algorithm pseudocode and design
- **[Workflow Doc](docs/workflow_doc.md)** — Runtime pipeline and configuration guide

---

## Credits

Cryptanalysis approach based on the research and algorithms developed by **George Lasry**.
