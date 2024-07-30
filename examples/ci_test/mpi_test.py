import madupite

def main():
    instance = madupite.initialize_madupite()
    rank, size = madupite.mpi_rank_size()
    print(f"Rank: {rank}, Size: {size}")


if __name__ == "__main__":
    main()