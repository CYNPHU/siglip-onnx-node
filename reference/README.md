# Reference images

Folder structure **is** the label:

```
reference/
  <brand>/
    <sku>/
      1.jpg
      2.jpg      ← 3+ images per SKU from different angles/lighting helps a lot
```

Example:

```
reference/acme/shampoo-blue/1.jpg
reference/acme/shampoo-blue/2.jpg
reference/acme/shampoo-green/1.jpg
reference/other-brand/soap-bar/1.jpg
```

Images are gitignored — this index is meant to be built from your own photos:

```bash
npm run build-index
```
