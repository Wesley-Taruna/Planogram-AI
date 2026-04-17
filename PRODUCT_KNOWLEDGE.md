# Product Knowledge Base — FamilyMart Indonesia Snack Aisle

> This file is loaded by `vision_checker.py` and injected into every Gemini
> vision prompt. Its purpose is to give the AI a **stable reference** so it
> reads the same product the same way every time. Edit this file whenever you
> observe a repeated mis-read; the AI will pick up the correction on the very
> next run.

---

## 1. Language map (Indonesian ↔ English)

Indonesian labels and planogram text mix freely. These are **equivalent** —
always treat them as the same word when comparing detections vs. planograms:

| Indonesian on pack | English / planogram term |
|---|---|
| Rasa | (flavor connector — ignore, it just means "flavor of") |
| Cokelat / Coklat | Chocolate |
| Karamel | Caramel |
| Keju | Cheese |
| Ayam Bakar | Grilled Chicken / Chicken |
| Ayam Goreng | Fried Chicken (DO NOT confuse with Ayam Bakar) |
| Kaldu Ayam | Chicken Broth / Cheeky Chicken |
| Jagung Bakar | Roasted Corn |
| Jagung | Corn |
| Sambal Balado | Chili Balado |
| Pedas | Spicy |
| Pedas Manis | Sweet Spicy |
| Mie / Mi | Noodle (both spellings mean the same thing) |
| Mie Goreng | Fried Noodle |
| Singkong | Cassava (it's just the raw material — not identifying) |
| Kripik / Keripik | Chips (category word — not identifying) |
| Snack | Snack (category word — not identifying) |
| Rumput Laut | Seaweed |
| Cabai / Cabe | Chili |
| Bawang | Onion |
| Manis | Sweet |
| Original | Original / Plain |

---

## 2. Brand spelling variants (same brand, different on-pack text)

- `Chuba` ≡ `Chuba Keripik Singkong` (brand name, not a flavor)
- `Qtela` = `Qtela Singkong` (the "Singkong" just repeats "cassava")
- `Kusuka` = `Kusuka Kripik Singkong` = `Kusuka Keripik Singkong`
- `Kobe Bonkrispi` ≈ `Bon Crispy Krispi` (alternate on-pack spelling)
- `Cem Cem` = the brand for Pop Cornz / Corn Shots variants
- `Gemez Enaak` = `Gemez Mi Enaak` = `Enaak Snack Mi` (all same snack noodle)
- `Spix` = `Spix Mi` (noodle-snack brand)
- `Happytos` — red bag = Corn Chips, green bag = Tortilla Chips
- `Dua Kelinci` — parent brand, `Tos Tos` is their tortilla line
- `Wonhae Topokki` = `Topokki` (Korean rice-cake snack brand)
- `Mie Kremezz` = `Mi Kremezz`
- `Chitato` vs `Chiki` — **different brands**, both from Indofood. Don't confuse.

---

## 3. Common visual confusion pairs — READ LABELS CAREFULLY

These pairs look alike on the shelf but are **different products**. If you
see one, double-check the flavor word before reporting:

| Looks similar | Different because |
|---|---|
| Kusuka Keju Bakar (orange bag) | Kusuka Barbeque (brown bag) |
| Kusuka Barbeque | Kusuka Balado (red) |
| Qtela Balado | Qtela Original (yellow vs red) |
| Cheetos Cheese | Cheetos Roasted Corn & Cheese |
| Chiki Balls Cheese | Chiki Balls Cheeky Chicken |
| Chiki Balls Coklat | Chiki Balls Curious Choco |
| Chiki Twist Keju Duo | Chiki Twist Roasted Corn |
| Oishi Pillows Coklat | Oishi Pillows Ubi |
| Cem Cem Corn Shot Ayam Goreng | Cem Cem Corn Shot Cokelat |
| Taro Tempe Sambal Matah | Taro Tempe Himalayan Salt |
| Mister Potato Crisps Roasted Corn | Momogi Stick Jagung Bakar (totally different brands) |

---

## 4. Packaging color → flavor cues (Indonesian snacks)

Use **color** to disambiguate when the flavor text is unreadable:

### Chuba / Kusuka / Qtela (cassava chips)
- **Yellow bag** → Keju (Cheese) / Original
- **Red / orange bag** → Balado / Sambal Balado
- **Brown / dark bag** → Barbeque / BBQ / Coklat
- **Green bag** → Keju Bakar / Nori / Herb
- **Black bag** → Ayam Lada Hitam / Black Pepper / Charcoal

### Chiki Balls (corn-puff balls)
- **Yellow** → Cheese / Keju
- **Brown** → Coklat (Chocolate)
- **Green** → Kaldu Ayam (Chicken Broth) / Cheeky Chicken
- **Red** → Spicy

### Cheetos / Momogi / Twistko (corn snacks)
- **Orange** → Cheese Pck
- **Yellow** → Roasted Corn & Cheese / Jagung Bakar Keju
- **Red** → Balado / Spicy
- **Green** → Barbeque

### Oishi Pillows / Pop Corn / Sponge Crunch
- **Brown** → Coklat / Chocolate
- **Yellow / Gold** → Belgian Butter / Caramel
- **Purple** → Ubi (sweet potato)
- **Red** → Strawberry

### Taro / Tempe snacks
- **Blue** → Himalayan Salt
- **Red** → Sambal Matah / Cabai Rawit
- **Green** → Rumput Laut (Seaweed)
- **Yellow** → Jagung / Cheese

### Tos Tos / Happytos
- **Red** → Corn Chips original (Happytos Merah)
- **Green** → Tortilla Chips original (Happytos Hijau)
- **Yellow** → Tortilla Original
- **Orange** → Nacho Cheese
- **Dark red** → Korean BBQ

### Mie noodle snacks (Gemez, Spix, Mi Kremezz)
- **Red** → Mi Goreng / Enaak
- **Yellow** → Soba Chicken / Ayam Bakar
- **Orange** → Sambal Balado

---

## 5. Canonical product catalog (snippet — expand as needed)

Known planogram products, with the **exact** canonical name. If you see one of
these on the shelf and the label is partially unreadable, prefer the canonical
name over a guess.

### Shelf 1 (cassava & tortilla chips)
- Chuba Keripik Singkong Keju 125g (yellow)
- Chuba Cassava Sambal Balado 125g (red)
- Kusuka Keripik Singkong Barbeque 180g (brown)
- Kusuka Kripik Singkong Keju Bakar 180g (orange)
- Kusuka Kripik Singkong Balado 180g (red)
- Qtela Singkong Balado 100g (red)
- Qtela Singkong Original 175g (yellow)
- Qtela Balado 175g (red)
- Qtela Singkong Barbeque 60g (brown)
- Kusuka Kripik Singkong Super Pedas 50g (red)
- Kusuka Kripik Singkong Bbq 60g (brown)
- Kusuka Kripik Singkong Ayam Lada Hitam 60g (black)
- Kobe Bonkrispi Ubi Original Pck 50gr (yellow)
- Kobe Bonkrispi Ubi Balado Pck 50gr (red)
- Kusuka Krpk Singkong Jagung Amerika 100g (yellow)
- Kusuka Kripik Singkong Original 180g (green)
- Taro Tempe Himalayan Salt 50g (blue)
- Taro Tempe Sambal Matah 50g (red)
- Garuda Corn Tornado Sea Salt Pck 70g (blue)
- Mr. Hottest Maitos Sambal Balado 140g (red)
- Maxicorn Roasted Corn 140gr (yellow)
- Maxicorn Nacho Cheese 140gr (orange)
- Happytos Merah Corn Chips 140G (red)
- Happytos Hijau Tortilla Chips 140G (green)
- Dua Kelinci Tos Tos Original Pck 175g (yellow)
- Dua Kelinci Tos Tos Nacho Cheese 140g (orange)
- Dua Kelinci Tos Tos Korean BBQ 140g (dark red)

### Shelf 4 (corn puffs, twists, popcorn, mie snacks)
- Chiki Balls Cheese 200g (yellow)
- Chiki Balls Coklat 200g (brown)
- Cheetos Cheese Pck 120gr (orange)
- Cheetos Roasted Corn & Cheese Pck 120gr (yellow)
- Twistko Snack Jagung Bakar 145g (red)
- Smax Ring Rasa Keju 40g (yellow)
- Momogi Classic Series 108 Gr (blue)
- Momogi Mini Stick Bite Roasted Corn 50g (yellow)
- Chiki Balls Cheese 55g (yellow small)
- Chiki Twist Roasted Corn Pck 120g (red)
- Jetz Choco Fiesta Coklat 65g (blue)
- Smax Ball Coklat 40g (brown)
- Momogi Star Bite Chocolate 50g (brown)
- Chiki Balls Coklat Pck 50g (brown)
- Chiki Balls Kaldu Ayam Pck 50g (green)
- Cheetos Cheese Pck 60gr (orange)
- Chiki Twist Roasted Corn 75g (red)
- Chiki Twist Keju Duo 70g (yellow)
- Chiki Twist Flaming Hot 75g (red)
- Chiki Twist Jagung Bakar Pedas Manis 65g (orange)
- Chiki Puffs Cheddar Cheese 60g (yellow)
- Twistko Snack Jagung Bakar 70g (red)
- Cem Cem Corn Shot Cokelat 55g (brown)
- Cem Cem Corn Shot Ayam Goreng 55g (orange)
- Cem Cem Pop Corn Chocolate 55g (brown)
- Oishi Pop Corn Caramel 100g (yellow)
- Oishi Pop Corn Belgian Butter 100gr (yellow/gold)
- Oishi Sponge Crunch Coklat 100g (brown)
- Oishi Pillows Coklat 100g (brown)
- Oishi Pillows Ubi 100g (purple)
- Gemez Mie Enaak 22g (red)
- Spix Soba Mie Chicken 21g (yellow)
- Spix Mie Goreng 50g (red)
- Mie Kremezz Rasa Mie Goreng 30G (orange)
- Wonhae Topokki Snack Cheese Buldak Pck 80gr (red)
- Wonhae Topokki Snack Creamy Rose 80g (pink)
- Chiki Popcorn Salted Caramel Pck 72gr (yellow)

---

## 6. Hard rules for the AI

1. **Never invent a product.** If you can't read the label and nothing in this
   catalog matches what you see, say `needs_web_check: true`.
2. **Match the flavor word exactly.** "Keju Bakar" ≠ "Barbeque". "Coklat" ≠
   "Caramel". Flavor words are the primary discriminator.
3. **Brand prefix matters.** "Kobe Bonkrispi Ubi" is NOT "Kusuka Singkong
   Black Pepper" even if the bag color is similar.
4. **If a product is behind another, don't skip it.** Report it with lower
   confidence and `ambiguity_note: "partially occluded by front item"`.
5. **Weight/size is not identifying.** Ignore "100g", "50gr", "Pck" when
   deciding which product it is — the flavor + brand determine the product.
6. **"Rasa X" always means "X flavor".** Strip "Rasa" mentally before naming
   the product. "Corn Shots Rasa Cokelat" = "Corn Shots Chocolate".
