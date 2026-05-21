import { Module } from '@nestjs/common';
import { BookService } from './book.service';
import { BookController } from './book.controller';
import type { Book, BookRepository } from './entities/book.entity';

@Module({
  controllers: [BookController],
  providers: [
    BookService,
    {
      provide: 'BOOK_REPOSITORY',
      useFactory: (): BookRepository => {
        const books: Book[] = [
          { id: 1, name: 'Book 1' },
          { id: 2, name: 'Book 2' },
        ];

        return {
          findAll: () => books,
        };
      },
    },
  ],
})
export class BookModule {}
