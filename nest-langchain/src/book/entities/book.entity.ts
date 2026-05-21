export interface Book {
  id: number;
  name: string;
}

export interface BookRepository {
  findAll(): Book[];
}
