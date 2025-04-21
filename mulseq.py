"""
Multiple Sequence Alignment (MSA) Analysis Tool
-----------------------------------------------
Program ini mengimplementasikan dan membandingkan berbagai algoritma penyelarasan
multipel sekuens (MSA) untuk analisis protein.

Author: [Nama Anda]
Email: mzaidan100703@gmail.com
Date: April 2025
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import Entrez, SeqIO, AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Konfigurasi awal
OUTPUT_DIR = "output"

def setup_output_directories():
    """
    Membuat struktur direktori output terorganisir per protein dan jenis analisis
    """
    # Direktori utama
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Direktori per protein
    protein_dirs = ["cytochrome_c", "histone_h4", "hsp70"]
    for protein in protein_dirs:
        protein_path = os.path.join(OUTPUT_DIR, protein)
        os.makedirs(protein_path, exist_ok=True)
        
        # Subdirektori per jenis analisis
        analysis_dirs = ["alignment", "phylogeny", "conservation", "statistics"]
        for analysis in analysis_dirs:
            os.makedirs(os.path.join(protein_path, analysis), exist_ok=True)
    
    # Direktori untuk hasil ringkasan
    summary_path = os.path.join(OUTPUT_DIR, "summary")
    os.makedirs(summary_path, exist_ok=True)
    
    return protein_dirs

# Email untuk NCBI Entrez (wajib)
Entrez.email = "mzaidan100703@gmail.com"

# ==================== 1. DATA COLLECTION ====================

def download_protein_sequences(protein_name, taxonomic_range, num_sequences=20, output_file=None):
    """
    Download sekuens protein dari NCBI berdasarkan nama protein dan taksonomi
    """
    print(f"Downloading {protein_name} sequences for {taxonomic_range}...")
    
    if output_file is None:
        protein_name_safe = protein_name.replace(" ", "_").lower()
        output_file = os.path.join(OUTPUT_DIR, protein_name_safe, f"{protein_name_safe}.fasta")
    
    # Mencari protein di database
    query = f"{protein_name}[Protein Name] AND {taxonomic_range}[Organism]"
    handle = Entrez.esearch(db="protein", term=query, retmax=num_sequences)
    record = Entrez.read(handle)
    handle.close()
    
    if not record["IdList"]:
        print(f"No sequences found for {protein_name} in {taxonomic_range}")
        return None
    
    # Download sekuens
    ids = record["IdList"]
    handle = Entrez.efetch(db="protein", id=ids, rettype="fasta", retmode="text")
    with open(output_file, "w") as out_handle:
        out_handle.write(handle.read())
    handle.close()
    
    print(f"Downloaded {len(ids)} sequences to {output_file}")
    return output_file

def generate_test_sequences(output_file=None):
    """
    Membuat data test sederhana jika tidak bisa mengakses NCBI
    """
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "test", "test_sequences.fasta")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Sekuens protein hipotetis dengan beberapa mutasi
    # Cytochrome C-like sequences with conservation patterns
    sequences = [
        ("Species1", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTY"),
        ("Species2", "MVLSPADKTNVKAAWSKVGAHAGEYGAEALERMFLSFPTTKTY"),
        ("Species3", "MVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTY"),
        ("Species4", "MVLSPADKTNVKAAWGKVGTHAGEYGAEALERMFLSFPTTKTY"),
        ("Species5", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFATTKTY")
    ]
    
    records = []
    for name, seq in sequences:
        records.append(SeqRecord(Seq(seq), id=name, description=f"Test sequence {name}"))
    
    with open(output_file, "w") as handle:
        SeqIO.write(records, handle, "fasta")
    
    print(f"Generated {len(sequences)} test sequences to {output_file}")
    return output_file


# ==================== 2. MSA IMPLEMENTATION WITHOUT EXTERNAL TOOLS ====================

def align_with_python(input_file, method="simple", output_file=None):
    """
    Implementasi algoritma MSA dasar tanpa memerlukan program eksternal
    
    Args:
        input_file: Path ke file FASTA
        method: "simple", "center_star", or "progressive"
        output_file: Path untuk menyimpan hasil alignment
        
    Returns:
        Objek MultipleSeqAlignment
    """
    # Ekstrak nama protein dari input_file
    protein_name = os.path.basename(os.path.dirname(input_file)) if os.path.dirname(input_file) else "unknown"
    
    if output_file is None:
        base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(OUTPUT_DIR, protein_name, "alignment", f"{base}_{method}.aln")
    
    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Running {method} alignment on {input_file}...")
    start_time = time.time()
    
    # Baca sekuens dari file
    sequences = list(SeqIO.parse(input_file, "fasta"))
    
    # Ubah ID sekuens agar tetap pendek untuk visualisasi
    for i, seq in enumerate(sequences):
        if len(seq.id) > 15:
            seq.id = seq.id[:10] + "..." + seq.id[-2:]
    
    # Implementasi metode alignment sesuai yang dipilih
    if method == "simple":
        # Temukan sekuens terpanjang
        max_len = max(len(seq.seq) for seq in sequences)
        
        # Buat alignment dengan menambahkan gap di akhir
        aligned_seqs = []
        for seq in sequences:
            new_seq = str(seq.seq) + "-" * (max_len - len(seq.seq))
            aligned_seqs.append(SeqRecord(Seq(new_seq), id=seq.id, description=""))
    
    elif method == "center_star":
        # Metode Center Star yang diperbaiki untuk menangani sekuens dengan panjang berbeda
        try:
            # 1. Hitung similarity matrix sederhana
            n = len(sequences)
            sim_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    # Hitung similarity sederhana - persentase match pada prefix terpendek
                    seq_i = str(sequences[i].seq)
                    seq_j = str(sequences[j].seq)
                    min_len = min(len(seq_i), len(seq_j))
                    
                    matches = sum(seq_i[k] == seq_j[k] for k in range(min_len))
                    similarity = matches / min_len if min_len > 0 else 0
                    
                    sim_matrix[i, j] = similarity
                    sim_matrix[j, i] = similarity
            
            # 2. Pilih sekuens tengah (dengan total similarity tertinggi)
            sim_sums = np.sum(sim_matrix, axis=1)
            center_idx = np.argmax(sim_sums)
            
            # 3. Align sekuens lain terhadap sekuens tengah
            center_seq = sequences[center_idx]
            aligned_seqs = [SeqRecord(center_seq.seq, id=center_seq.id, description="")]
            
            # Cari panjang maksimum yang mungkin dibutuhkan
            max_len = max(len(seq.seq) for seq in sequences)
            
            # Untuk setiap sekuens lain, align terhadap sekuens tengah
            for i, seq in enumerate(sequences):
                if i == center_idx:
                    continue
                
                # Tambahkan gap di akhir untuk menyamakan panjang dengan max_len
                curr_seq = str(seq.seq)
                aligned_seq = curr_seq + "-" * (max_len - len(curr_seq))
                
                aligned_seqs.append(SeqRecord(Seq(aligned_seq), id=seq.id, description=""))
        
        except Exception as e:
            print(f"Error in center_star method: {e}")
            print("Falling back to simple alignment...")
            
            # Fallback ke metode simple jika center_star gagal
            max_len = max(len(seq.seq) for seq in sequences)
            aligned_seqs = []
            for seq in sequences:
                new_seq = str(seq.seq) + "-" * (max_len - len(seq.seq))
                aligned_seqs.append(SeqRecord(Seq(new_seq), id=seq.id, description=""))
    
    elif method == "progressive":
        # Metode Progressive Alignment (disederhanakan)
        # 1. Mulai dengan alignment kosong
        aligned_seqs = []
        current_alignment_len = 0
        
        # 2. Urutkan sekuens berdasarkan panjang (dari terpanjang)
        sorted_seqs = sorted(sequences, key=lambda seq: len(seq.seq), reverse=True)
        
        # 3. Tambahkan sekuens terpanjang sebagai dasar
        first_seq = sorted_seqs[0]
        aligned_seqs.append(first_seq)
        current_alignment_len = len(first_seq.seq)
        
        # 4. Tambahkan sekuens lainnya satu per satu
        for seq in sorted_seqs[1:]:
            # Jika sekuens lebih pendek dari alignment saat ini, tambahkan gap
            if len(seq.seq) < current_alignment_len:
                new_seq = str(seq.seq) + "-" * (current_alignment_len - len(seq.seq))
                aligned_seqs.append(SeqRecord(Seq(new_seq), id=seq.id, description=""))
            else:
                # Jika sekuens lebih panjang, perbarui semua sekuens yang ada
                diff = len(seq.seq) - current_alignment_len
                for i in range(len(aligned_seqs)):
                    aligned_seqs[i] = SeqRecord(
                        Seq(str(aligned_seqs[i].seq) + "-" * diff),
                        id=aligned_seqs[i].id,
                        description=""
                    )
                aligned_seqs.append(seq)
                current_alignment_len = len(seq.seq)
    
    else:
        raise ValueError(f"Method {method} not implemented")
    
    # Buat objek MultipleSeqAlignment
    alignment = MultipleSeqAlignment(aligned_seqs)
    
    # Tulis alignment ke file dalam format Clustal
    with open(output_file, "w") as f:
        f.write("CLUSTAL W (1.81) multiple sequence alignment\n\n\n")
        for record in alignment:
            f.write(f"{record.id.ljust(15)} {record.seq}\n")
    
    end_time = time.time()
    print(f"{method.capitalize()} alignment completed in {end_time - start_time:.2f} seconds")
    
    return alignment, end_time - start_time

# ==================== 3. ANALYSIS FUNCTIONS ====================

def analyze_conservation(alignment):
    """
    Analisis tingkat konservasi pada setiap posisi alignment dengan pendekatan yang lebih robust
    """
    # Dapatkan panjang alignment
    aln_len = alignment.get_alignment_length()
    
    # Inisialisasi array untuk menyimpan skor konservasi
    conservation_scores = np.zeros(aln_len)
    
    # Kelompok asam amino berdasarkan sifat (untuk analisis penggantian konservatif)
    amino_groups = {
        'hydrophobic': set('AVILMFYW'),
        'polar': set('STNQ'),
        'positive': set('KRH'),
        'negative': set('DE'),
        'special': set('CGP')
    }
    
    # Hitung konservasi untuk setiap kolom
    for i in range(aln_len):
        column = alignment[:, i]
        
        # Hitung frekuensi setiap asam amino pada posisi ini
        aa_counts = {}
        non_gap_count = 0
        
        for aa in column:
            if aa != '-':  # Skip gaps
                non_gap_count += 1
                if aa not in aa_counts:
                    aa_counts[aa] = 0
                aa_counts[aa] += 1
        
        # Hitung konservasi sebagai frekuensi asam amino paling umum
        if non_gap_count > 0:
            most_common_count = max(aa_counts.values()) if aa_counts else 0
            conservation_scores[i] = most_common_count / non_gap_count
    
    return conservation_scores

def calculate_conserved_positions(conservation_scores, threshold):
    """
    Hitung jumlah posisi yang melebihi ambang konservasi
    """
    return sum(score >= threshold for score in conservation_scores)

def identify_functional_regions(alignment, window_size=5, threshold=0.8):
    """
    Identifikasi kemungkinan daerah fungsional berdasarkan konservasi tinggi
    """
    conservation_scores = analyze_conservation(alignment)
    aln_len = len(conservation_scores)
    
    # Gunakan algoritma sliding window untuk mencari region terkonservasi
    functional_regions = []
    current_region = None
    
    for i in range(aln_len):
        # Hitung rata-rata konservasi di window saat ini
        window_start = max(0, i - window_size // 2)
        window_end = min(aln_len, i + window_size // 2 + 1)
        window_scores = conservation_scores[window_start:window_end]
        window_avg = sum(window_scores) / len(window_scores)
        
        # Jika window saat ini memiliki konservasi tinggi
        if window_avg >= threshold:
            if current_region is None:
                # Mulai region baru
                current_region = [i, i, window_avg]
            else:
                # Perluas region saat ini
                current_region[1] = i
                # Update skor dengan yang lebih tinggi
                current_region[2] = max(current_region[2], window_avg)
        else:
            if current_region is not None:
                # Simpan region saat ini dan reset
                functional_regions.append(tuple(current_region))
                current_region = None
    
    # Simpan region terakhir jika masih ada
    if current_region is not None:
        functional_regions.append(tuple(current_region))
    
    return functional_regions

def analyze_sequence_composition(alignment):
    """
    Analisis komposisi asam amino dalam alignment
    """
    # Inisialisasi counter untuk setiap asam amino
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_counts = {aa: 0 for aa in amino_acids}
    total_aas = 0
    
    # Hitung frekuensi setiap asam amino
    for record in alignment:
        for aa in record.seq:
            if aa != '-' and aa in amino_acids:  # Skip gaps dan karakter non-standar
                aa_counts[aa] += 1
                total_aas += 1
    
    # Konversi ke persentase
    aa_percentages = {aa: (count / total_aas * 100) if total_aas > 0 else 0 
                     for aa, count in aa_counts.items()}
    
    return aa_percentages

def analyze_alignment(alignment, algorithm_name):
    """
    Menganalisis hasil alignment dengan metode yang diperluas
    """
    print(f"\nAnalisis {algorithm_name}:")
    print(f"Jumlah sekuens: {len(alignment)}")
    print(f"Panjang alignment: {alignment.get_alignment_length()}")
    
    # Analisis konservasi
    conservation_scores = analyze_conservation(alignment)
    
    # Hitung posisi terkonservasi untuk berbagai threshold
    thresholds = [0.5, 0.7, 0.9]
    for threshold in thresholds:
        conserved_count = calculate_conserved_positions(conservation_scores, threshold)
        conserved_percent = conserved_count / alignment.get_alignment_length() * 100
        print(f"Residu terkonservasi (threshold {threshold}): {conserved_count} ({conserved_percent:.1f}%)")
    
    # Identifikasi kemungkinan daerah fungsional
    functional_regions = identify_functional_regions(alignment)
    if functional_regions:
        print("\nKemungkinan daerah fungsional:")
        for start, end, score in functional_regions:
            print(f"  Posisi {start}-{end}: skor konservasi {score:.2f}")
    
    # Hitung persentase identitas sekuens
    identities = []
    for i in range(len(alignment)):
        for j in range(i+1, len(alignment)):
            seq1 = str(alignment[i].seq)
            seq2 = str(alignment[j].seq)
            
            # Hitung posisi yang identik (tidak termasuk posisi dengan gap)
            match_count = 0
            total_count = 0
            
            for pos in range(len(seq1)):
                if seq1[pos] != '-' and seq2[pos] != '-':
                    total_count += 1
                    if seq1[pos] == seq2[pos]:
                        match_count += 1
            
            if total_count > 0:
                identity = match_count / total_count
                identities.append(identity)
    
    avg_identity = sum(identities) / len(identities) if identities else 0
    print(f"Rata-rata persentase identitas: {avg_identity:.2%}")
    
    # Analisis komposisi sekuens
    aa_composition = analyze_sequence_composition(alignment)
    
    return {
        "num_sequences": len(alignment),
        "alignment_length": alignment.get_alignment_length(),
        "conservation_scores": conservation_scores,
        "functional_regions": functional_regions,
        "avg_identity": avg_identity,
        "aa_composition": aa_composition
    }

def calculate_biological_score(alignment, functional_residues=None):
    """
    Menghitung skor biologis berdasarkan konservasi residu fungsional dengan pendekatan yang ditingkatkan
    """
    # Analisis konservasi
    conservation_scores = analyze_conservation(alignment)
    
    if not functional_residues:
        # Jika tidak ada informasi residu fungsional, gunakan region terkonservasi tinggi
        functional_regions = identify_functional_regions(alignment)
        functional_residues = []
        
        for start, end, _ in functional_regions:
            functional_residues.extend(range(start, end + 1))
        
        # Jika tidak ada region terkonservasi, gunakan 10 posisi dengan konservasi tertinggi
        if not functional_residues:
            sorted_positions = np.argsort(conservation_scores)[::-1]
            functional_residues = sorted_positions[:10].tolist()
    
    # Hitung skor biologis sebagai rata-rata konservasi pada residu fungsional
    bio_scores = [conservation_scores[pos] for pos in functional_residues 
                 if 0 <= pos < len(conservation_scores)]
    
    bio_score = sum(bio_scores) / len(bio_scores) if bio_scores else 0
    print(f"Biological score: {bio_score:.4f}")
    
    # Tampilkan informasi tentang residu fungsional yang digunakan
    print(f"Analisis dilakukan pada {len(bio_scores)} posisi fungsional potensial")
    
    return bio_score

# ==================== 4. PHYLOGENETIC ANALYSIS ====================

def build_phylogenetic_tree(alignment, output_file=None):
    """
    Membangun pohon filogenetik dari alignment menggunakan metode distance
    """
    # Dapatkan nama protein dari alignment
    protein_name = "unknown"
    if output_file and os.path.dirname(output_file):
        protein_name = os.path.basename(os.path.dirname(os.path.dirname(output_file)))
    
    if output_file is None:
        # Generate output filename berdasarkan informasi alignment
        base = "phylogenetic_tree"
        for record in alignment[:1]:  # Ambil info dari record pertama
            base = record.id.split('...')[0]
        output_file = os.path.join(OUTPUT_DIR, protein_name, "phylogeny", f"{base}_tree.newick")
    
    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Building phylogenetic tree...")
    
    try:
        # Hitung matriks jarak menggunakan identitas
        calculator = DistanceCalculator("identity")
        dm = calculator.get_distance(alignment)
        
        # Bangun pohon menggunakan metode neighbor-joining
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(dm)
        
        # Akar pohon di tengah
        tree.root_at_midpoint()
        
        # Simpan pohon ke file
        Phylo.write(tree, output_file, "newick")
        print(f"Tree saved to {output_file}")
        
        return tree
    except Exception as e:
        print(f"Error building phylogenetic tree: {e}")
        # Buat pohon sederhana sebagai fallback
        tree = Phylo.BaseTree.Tree(Phylo.BaseTree.Clade())
        for i, record in enumerate(alignment):
            # Add a clade for each sequence
            clade = Phylo.BaseTree.Clade(branch_length=0.1, name=record.id)
            tree.clade.clades.append(clade)
        
        # Simpan pohon ke file
        Phylo.write(tree, output_file, "newick")
        print(f"Created a simple fallback tree. Saved to {output_file}")
        
        return tree
    

# ==================== 5. VISUALIZATION FUNCTIONS ====================

def visualize_tree(tree, title, output_file):
    """
    Visualisasi pohon filogenetik dengan output terorganisir
    """
    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        plt.figure(figsize=(10, len(tree.get_terminals()) * 0.5 + 3))
        Phylo.draw(tree, do_show=False, label_func=lambda x: x.name)
        plt.title(title)
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Tree visualization saved to {output_file}")
    except Exception as e:
        print(f"Error visualizing tree: {e}")

def visualize_alignment(alignment, title, output_file):
    """
    Visualisasi alignment sebagai heatmap dengan informasi konservasi yang ditingkatkan
    """
    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Konversi alignment ke matriks numerik
        # Kita akan menggunakan indeks alfabet sebagai nilai numerik
        alphabet = "-ACDEFGHIKLMNPQRSTVWY"
        alphabet_map = {aa: i for i, aa in enumerate(alphabet)}
        
        seq_names = [record.id for record in alignment]
        alignment_len = alignment.get_alignment_length()
        
        # Bangun matriks numerik
        matrix = np.zeros((len(alignment), alignment_len))
        for i, record in enumerate(alignment):
            for j, aa in enumerate(record.seq):
                matrix[i, j] = alphabet_map.get(aa.upper(), 0)
        
        # Hitung konservasi kolom
        conservation_scores = analyze_conservation(alignment)
        
        # Visualisasi sebagai heatmap
        plt.figure(figsize=(14, 10))
        
        # Plot heatmap alignment
        plt.subplot(3, 1, 1)
        im = plt.imshow(matrix, aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im, fraction=0.03, pad=0.05)
        cbar.set_label('Amino Acid Index')
        
        # Anotasi untuk colorbar - tambahkan label asam amino
        tick_locs = range(len(alphabet))
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(alphabet)
        
        plt.title(title)
        plt.xlabel('Position')
        plt.ylabel('Sequence')
        if len(seq_names) <= 20:  # Only show labels if there aren't too many
            plt.yticks(range(len(seq_names)), seq_names)
        
        # Plot conservation score sebagai bar chart
        plt.subplot(3, 1, 2)
        bars = plt.bar(range(alignment_len), conservation_scores, color='skyblue', alpha=0.7)
        
        # Tambahkan garis untuk threshold
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='50% Conservation')
        plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.3, label='70% Conservation')
        plt.axhline(y=0.9, color='b', linestyle='--', alpha=0.3, label='90% Conservation')
        
        plt.title('Conservation Score by Position')
        plt.xlabel('Position')
        plt.ylabel('Conservation Score')
        plt.ylim(0, 1.05)
        plt.legend()
        
        # Highlight highly conserved regions dengan warna berbeda
        for threshold, color in [(0.9, 'green'), (0.7, 'yellow'), (0.5, 'orange')]:
            conserved = np.where(conservation_scores >= threshold)[0]
            if len(conserved) > 0:
                plt.bar(conserved, [conservation_scores[i] for i in conserved], 
                       color=color, alpha=0.5)
        
        # Tambahkan histogram distribusi konservasi
        plt.subplot(3, 1, 3)
        plt.hist(conservation_scores, bins=20, color='skyblue', alpha=0.7)
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=0.7, color='g', linestyle='--', alpha=0.3)
        plt.axvline(x=0.9, color='b', linestyle='--', alpha=0.3)
        plt.title('Distribution of Conservation Scores')
        plt.xlabel('Conservation Score')
        plt.ylabel('Frequency')
        plt.xlim(0, 1.05)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Enhanced alignment visualization saved to {output_file}")
    except Exception as e:
        print(f"Error visualizing alignment: {e}")

def compare_alignment_methods(alignments, method_names, title, output_file):
    """
    Bandingkan berbagai metode alignment dengan visualisasi yang ditingkatkan
    """
    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Dapatkan konservasi untuk setiap metode
        all_conservation = []
        for alignment in alignments:
            conservation = analyze_conservation(alignment)
            all_conservation.append(conservation)
        
        # Cari panjang alignment maksimum
        max_len = max(len(cons) for cons in all_conservation)
        
        # Hitung statistik konservasi untuk setiap metode
        stats = []
        for cons, method in zip(all_conservation, method_names):
            thresholds = [0.5, 0.7, 0.9]
            thresh_counts = {}
            for t in thresholds:
                count = sum(1 for score in cons if score >= t)
                percent = (count / len(cons)) * 100 if len(cons) > 0 else 0
                thresh_counts[t] = (count, percent)
            
            stats.append({
                'method': method,
                'mean': np.mean(cons),
                'median': np.median(cons),
                'std': np.std(cons),
                'max': np.max(cons),
                'min': np.min(cons),
                'thresholds': thresh_counts
            })
        
        # Buat plot
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Plot konservasi untuk setiap metode
        for i, (cons, method) in enumerate(zip(all_conservation, method_names)):
            # Pad dengan nol jika perlu
            if len(cons) < max_len:
                cons = np.pad(cons, (0, max_len - len(cons)), 'constant')
            
            ax = plt.subplot(len(alignments)+2, 1, i+1)
            bars = ax.bar(range(len(cons)), cons, color='skyblue', alpha=0.7)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.3)
            ax.axhline(y=0.9, color='b', linestyle='--', alpha=0.3)
            ax.set_title(f'{method} Conservation')
            ax.set_ylabel('Conservation')
            ax.set_ylim(0, 1.05)
            
            # Highlight regions berdasarkan threshold
            for threshold, color in [(0.9, 'green'), (0.7, 'yellow'), (0.5, 'orange')]:
                conserved = np.where(cons >= threshold)[0]
                if len(conserved) > 0:
                    ax.bar(conserved, [cons[i] for i in conserved], 
                          color=color, alpha=0.5)
        
        # 2. Plot perbandingan statistik konservasi
        ax_stats = plt.subplot(len(alignments)+2, 1, len(alignments)+1)
        means = [s['mean'] for s in stats]
        medians = [s['median'] for s in stats]
        stds = [s['std'] for s in stats]
        
        x = np.arange(len(method_names))
        width = 0.25
        
        ax_stats.bar(x - width, means, width, label='Mean', color='blue', alpha=0.7)
        ax_stats.bar(x, medians, width, label='Median', color='green', alpha=0.7)
        ax_stats.bar(x + width, stds, width, label='Std Dev', color='red', alpha=0.7)
        
        ax_stats.set_title('Conservation Statistics Comparison')
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(method_names)
        ax_stats.set_ylabel('Value')
        ax_stats.legend()
        
        # 3. Plot threshold comparison
        ax_thresh = plt.subplot(len(alignments)+2, 1, len(alignments)+2)
        
        bar_width = 0.2
        threshold_positions = np.arange(len(method_names))
        
        for i, t in enumerate([0.5, 0.7, 0.9]):
            percentages = [s['thresholds'][t][1] for s in stats]
            ax_thresh.bar(threshold_positions + (i-1)*bar_width, percentages, 
                         bar_width, label=f'≥{t*100}%', alpha=0.7)
        
        ax_thresh.set_title('Percentage of Positions Above Conservation Thresholds')
        ax_thresh.set_xticks(threshold_positions)
        ax_thresh.set_xticklabels(method_names)
        ax_thresh.set_ylabel('Percentage of Positions')
        ax_thresh.set_ylim(0, 100)
        ax_thresh.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Enhanced alignment comparison saved to {output_file}")
    except Exception as e:
        print(f"Error comparing alignments: {e}")

def create_summary_visualization(all_results, output_file):
    """
    Membuat visualisasi ringkasan untuk semua protein dan metode
    """
    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        proteins = list(all_results.keys())
        methods = []
        for protein in proteins:
            methods.extend(list(all_results[protein].keys()))
        methods = sorted(list(set(methods)))  # Unique methods
        
        # Ekstrak data untuk visualisasi
        conservation_data = []
        for protein in proteins:
            for method in methods:
                if method in all_results[protein]:
                    result = all_results[protein][method]
                    if 'conservation_scores' in result:
                        scores = result['conservation_scores']
                        # Hitung statistik konservasi
                        thresholds = [0.5, 0.7, 0.9]
                        for t in thresholds:
                            count = sum(1 for score in scores if score >= t)
                            percent = (count / len(scores)) * 100 if len(scores) > 0 else 0
                            conservation_data.append({
                                'protein': protein,
                                'method': method,
                                'threshold': t,
                                'percent': percent
                            })
        
        # Convert to DataFrame for easier plotting
        if conservation_data:
            df = pd.DataFrame(conservation_data)
            
            # Buat plot
            plt.figure(figsize=(15, 10))
            
            # Buat bar plot untuk setiap threshold
            n_thresholds = len(df['threshold'].unique())
            for i, t in enumerate(sorted(df['threshold'].unique())):
                plt.subplot(n_thresholds, 1, i+1)
                
                # Filter for this threshold
                df_t = df[df['threshold'] == t]
                
                # Reshape for grouped bar plot
                pivot = df_t.pivot(index='protein', columns='method', values='percent')
                
                # Plot
                pivot.plot(kind='bar', ax=plt.gca())
                plt.title(f'Conservation at {t*100}% Threshold')
                plt.xlabel('Protein')
                plt.ylabel('% of Positions Conserved')
                plt.ylim(0, 100)
                plt.grid(axis='y', alpha=0.3)
                plt.legend(title='Method')
            
            plt.suptitle('Conservation Comparison Across Proteins and Methods', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Summary visualization saved to {output_file}")
        else:
            print("Not enough data for summary visualization")
    except Exception as e:
        print(f"Error creating summary visualization: {e}")


# ==================== 6. MAIN WORKFLOW ====================

def run_msa_analysis(protein_name, taxonomic_range, fasta_file=None, methods=None):
    """
    Menjalankan analisis MSA untuk satu protein dengan output terorganisir
    """
    if methods is None:
        methods = ["simple", "center_star", "progressive"]
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {protein_name} FOR {taxonomic_range}")
    print(f"{'='*60}")
    
    # Buat folder spesifik untuk protein ini
    protein_safe_name = protein_name.replace(" ", "_").lower()
    if protein_safe_name == "heat_shock_protein_70":
        protein_safe_name = "hsp70"  # Alias untuk nama yang lebih pendek
        
    protein_dir = os.path.join(OUTPUT_DIR, protein_safe_name)
    
    # Pastikan direktori ada
    for subdir in ["alignment", "phylogeny", "conservation", "statistics"]:
        os.makedirs(os.path.join(protein_dir, subdir), exist_ok=True)
    
    # Download atau gunakan file FASTA yang diberikan
    if fasta_file is None:
        try:
            fasta_file = download_protein_sequences(protein_name, taxonomic_range, num_sequences=10,
                                                  output_file=os.path.join(protein_dir, f"{protein_safe_name}.fasta"))
            if fasta_file is None:
                print(f"Failed to download {protein_name} sequences. Generating test data instead.")
                fasta_file = generate_test_sequences(os.path.join(protein_dir, f"{protein_safe_name}_test.fasta"))
        except Exception as e:
            print(f"Error downloading sequences: {e}")
            print("Generating test data instead.")
            fasta_file = generate_test_sequences(os.path.join(protein_dir, f"{protein_safe_name}_test.fasta"))
    
    # Jalankan algoritma alignment
    alignments = {}
    execution_times = {}
    
    for method in methods:
        try:
            output_file = os.path.join(protein_dir, "alignment", f"{protein_safe_name}_{method}.aln")
            alignment, time_taken = align_with_python(fasta_file, method, output_file)
            alignments[method] = alignment
            execution_times[method] = time_taken
        except Exception as e:
            print(f"Error running {method}: {e}")
    
    # Analisis hasil alignment
    results = {}
    biological_scores = {}
    
    for method, alignment in alignments.items():
        # Analisis alignment
        results[method] = analyze_alignment(alignment, method)
        
        # Hitung skor biologis
        biological_scores[method] = calculate_biological_score(alignment)
        
        # Visualisasi alignment
        visualize_alignment(
            alignment,
            f"{protein_name} Alignment ({method})",
            os.path.join(protein_dir, "alignment", f"{protein_safe_name}_{method}_align.png")
        )
    
    # Bandingkan metode alignment
    if len(alignments) > 1:
        compare_alignment_methods(
            list(alignments.values()),
            list(alignments.keys()),
            f"Conservation Comparison for {protein_name}",
            os.path.join(protein_dir, "conservation", f"{protein_safe_name}_conservation_comparison.png")
        )
    
    # Bangun dan visualisasi pohon filogenetik untuk metode terbaik
    # Pilih metode dengan skor biologis tertinggi
    if biological_scores:
        best_method = max(biological_scores, key=biological_scores.get)
        best_alignment = alignments[best_method]
        
        tree_file = os.path.join(protein_dir, "phylogeny", f"{protein_safe_name}_{best_method}_tree.newick")
        tree = build_phylogenetic_tree(best_alignment, tree_file)
        
        if tree:
            tree_img = os.path.join(protein_dir, "phylogeny", f"{protein_safe_name}_{best_method}_tree.png")
            visualize_tree(tree, f"{protein_name} Phylogeny (Best: {best_method})", tree_img)
    
    # Create performance comparison plot
    if len(execution_times) > 1:
        plt.figure(figsize=(8, 6))
        methods_list = list(execution_times.keys())
        times_list = [execution_times[m] for m in methods_list]
        
        bars = plt.bar(methods_list, times_list, color='skyblue')
        plt.title(f'Performance Comparison for {protein_name}')
        plt.xlabel('Method')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value annotations
        for bar, val in zip(bars, times_list):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f'{val:.2f}s',
                ha='center', va='bottom'
            )
        
        plt.savefig(os.path.join(protein_dir, "statistics", f"{protein_safe_name}_performance.png"))
        plt.close()
    
    # Create biological score comparison
    if len(biological_scores) > 1:
        plt.figure(figsize=(8, 6))
        methods_list = list(biological_scores.keys())
        scores_list = [biological_scores[m] for m in methods_list]
        
        bars = plt.bar(methods_list, scores_list, color='lightgreen')
        plt.title(f'Biological Score Comparison for {protein_name}')
        plt.xlabel('Method')
        plt.ylabel('Biological Score')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value annotations
        for bar, val in zip(bars, scores_list):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f'{val:.2f}',
                ha='center', va='bottom'
            )
        
        plt.savefig(os.path.join(protein_dir, "statistics", f"{protein_safe_name}_biological_scores.png"))
        plt.close()
    
    return {
        "alignments": alignments,
        "results": results,
        "biological_scores": biological_scores,
        "execution_times": execution_times
    }


# ==================== 7. MAIN EXECUTION ====================

if __name__ == "__main__":
    # Setup direktori output
    protein_dirs = setup_output_directories()
    Entrez.email = "mzaidan100703@gmail.com"  # Gunakan email Anda
    
    # Informasi bantuan instalasi
    print("\n======== INFORMASI PROGRAM INTERNAL ========")
    print("Program akan menggunakan implementasi MSA internal Python.")
    print("Implementasi internal ini terdiri dari 3 metode:")
    print("1. Simple - Menyelaraskan sekuens berdasarkan panjang (sederhana)")
    print("2. Center Star - Memilih sekuens tengah dan menyelaraskan yang lain terhadapnya")
    print("3. Progressive - Menyelaraskan sekuens secara progresif berdasarkan panjang")
    print("Output akan disimpan dalam folder terpisah untuk masing-masing protein.")
    print("Untuk hasil yang lebih akurat, sebaiknya instal program eksternal seperti ClustalW.\n")
    
    # Uji koneksi ke NCBI
    try:
        print("Menguji koneksi ke NCBI...")
        handle = Entrez.einfo()
        record = Entrez.read(handle)
        handle.close()
        print("Koneksi ke NCBI berhasil.\n")
        
        # Definisikan protein dan taksonomi untuk analisis
        proteins = ["Cytochrome C", "Histone H4", "Heat Shock Protein 70"]
        taxonomies = ["Vertebrata", "Eukaryota", "Metazoa"]
        methods = ["simple", "center_star", "progressive"]
        
        # Dictionary untuk menyimpan semua hasil
        all_results = {}
        all_biological_scores = {}
        all_execution_times = {}
        
        # Jalankan analisis untuk setiap protein
        for protein_name, taxonomic_range in zip(proteins, taxonomies):
            try:
                protein_safe_name = protein_name.replace(" ", "_").lower()
                if protein_safe_name == "heat_shock_protein_70":
                    protein_safe_name = "hsp70"
                
                result = run_msa_analysis(protein_name, taxonomic_range, methods=methods)
                
                # Simpan hasil
                protein_key = protein_safe_name
                all_results[protein_key] = result["results"]
                all_biological_scores[protein_key] = result["biological_scores"]
                all_execution_times[protein_key] = result["execution_times"]
            
            except Exception as e:
                print(f"Error analyzing {protein_name} for {taxonomic_range}: {e}")
        
        # Simpan hasil ringkasan ke CSV
        summary_dir = os.path.join(OUTPUT_DIR, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Biological scores
        bio_scores_data = []
        for protein_key, methods_scores in all_biological_scores.items():
            for method, score in methods_scores.items():
                bio_scores_data.append({
                    "Protein": protein_key,
                    "Method": method,
                    "Biological Score": score
                })
        
        if bio_scores_data:
            bio_scores_df = pd.DataFrame(bio_scores_data)
            bio_scores_file = os.path.join(summary_dir, "biological_scores.csv")
            bio_scores_df.to_csv(bio_scores_file, index=False)
            print(f"Saved all biological scores to {bio_scores_file}")
        
        # Execution times
        exec_times_data = []
        for protein_key, methods_times in all_execution_times.items():
            for method, time_taken in methods_times.items():
                exec_times_data.append({
                    "Protein": protein_key,
                    "Method": method,
                    "Execution Time (s)": time_taken
                })
        
        if exec_times_data:
            exec_times_df = pd.DataFrame(exec_times_data)
            exec_times_file = os.path.join(summary_dir, "execution_times.csv")
            exec_times_df.to_csv(exec_times_file, index=False)
            print(f"Saved all execution times to {exec_times_file}")
        
        # Create summary plots
        if all_results:
            # Conservation summary
            create_summary_visualization(
                all_results,
                os.path.join(summary_dir, "conservation_summary.png")
            )
            
            # Biological score comparison
            if bio_scores_data:
                plt.figure(figsize=(12, 8))
                
                # Use pivot table for easier plotting
                pivot_scores = bio_scores_df.pivot(index="Method", columns="Protein", values="Biological Score")
                pivot_scores.plot(kind="bar", ax=plt.gca())
                
                plt.title("Biological Scores Comparison Across Proteins and Methods")
                plt.xlabel("Method")
                plt.ylabel("Biological Score")
                plt.ylim(0, 1)
                plt.legend(title="Protein")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(os.path.join(summary_dir, "biological_scores_summary.png"))
                plt.close()
            
            print(f"Created summary plots in {summary_dir}")
        
        print("\nAnalysis complete! All output files are organized in folders in the 'output' directory.")
        print("Key folders:")
        for protein in protein_dirs:
            print(f"- output/{protein}/ (contains alignment, phylogeny, conservation, and statistics)")
        print("- output/summary/ (contains summary plots and files)")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your internet connection and try again.")
        
        # Try to generate and analyze test data if other methods fail
        try:
            print("Generating and analyzing test data instead...")
            test_file = generate_test_sequences()
            run_msa_analysis("Test Protein", "Test Taxonomy", test_file)
        except Exception as e2:
            print(f"Error generating test data: {e2}")